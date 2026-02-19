from __future__ import annotations

import collections
import csv
import importlib
import importlib.util
import os
from typing import Dict, List, Optional, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
import dill
import haiku as hk
import jax
import matplotlib
matplotlib.use('Agg')  # Headless environments.
import matplotlib.pyplot as plt
import numpy as np

from emergent_in_context_learning.experiment import experiment as experiment_lib
from emergent_in_context_learning.modules.embedding import InputEmbedder
from emergent_in_context_learning.modules.transformer import Transformer

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'analysis_config',
    None,
    'Python module path or filesystem path to the training config that matches '
    'the checkpoint (e.g., experiment/configs/images_all_exemplars.py).')
flags.DEFINE_string('checkpoint', None, 'Path to the checkpoint file or directory.')
flags.DEFINE_string('output_dir', '/tmp/common_context_umap', 'Directory for analysis artifacts.')
flags.DEFINE_integer('samples_per_class', 32, 'Samples collected per class for each condition.')
flags.DEFINE_integer('max_sampling_attempts', int(5e5), 'Safety bound on sampling attempts.')
flags.DEFINE_integer('sampling_seed', 0, 'Seed for numpy sampling.')
flags.DEFINE_integer('model_seed', 0, 'Seed for model stochasticity.')
flags.DEFINE_integer('umap_neighbors', 15, 'Number of neighbors for UMAP.')
flags.DEFINE_float('umap_min_dist', 0.4, 'Minimum distance parameter for UMAP.')
flags.DEFINE_enum(
    'projection_method', 'umap', ['umap', 'pca'],
    'Dimensionality reduction method to use when visualizing representations.')
flags.DEFINE_string(
    'metadata_npz',
    None,
    'Path to a previously saved metadata NPZ file. When provided, the script '
    'skips embedding collection and renders plots directly from the cached '
    'coordinates.')


def _load_training_config(config_path: str):
  """Loads the training config from either a module or filesystem path."""
  if not config_path:
    raise ValueError('The --analysis_config flag must be provided.')

  expanded_path = os.path.expanduser(config_path)
  module = None
  if expanded_path.endswith('.py') and os.path.exists(expanded_path):
    module_name = 'analysis_config_module'
    spec = importlib.util.spec_from_file_location(module_name, expanded_path)
    if spec is None or spec.loader is None:
      raise ImportError(f'Unable to load config from {expanded_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
  else:
    module_name = config_path
    if module_name.endswith('.py'):
      module_name = module_name[:-3]
    module_name = module_name.replace('/', '.').replace('\\', '.')
    module = importlib.import_module(module_name)

  if not hasattr(module, 'get_config'):
    raise ValueError(
        f'Config module "{config_path}" does not define a get_config() function.')

  config = module.get_config()
  exp_kwargs = getattr(config, 'experiment_kwargs', None)
  if exp_kwargs is None and isinstance(config, dict):
    exp_kwargs = config.get('experiment_kwargs')
  if exp_kwargs is None:
    raise ValueError(
        'The provided config must define experiment_kwargs with a nested config.')

  experiment_config = getattr(exp_kwargs, 'config', None)
  if experiment_config is None and isinstance(exp_kwargs, dict):
    experiment_config = exp_kwargs.get('config')
  if experiment_config is None:
    raise ValueError('experiment_kwargs must contain a "config" entry.')
  return experiment_config

def _load_checkpoint(path: str) -> Tuple[hk.Params, hk.State]:
  """Loads parameters and state from a checkpoint."""
  checkpoint_path = path
  if os.path.isdir(checkpoint_path):
    checkpoint_path = os.path.join(checkpoint_path, 'checkpoint.dill')
  if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
  with open(checkpoint_path, 'rb') as f:
    payload = dill.load(f)
  return payload['params'], payload['state']


def _build_forward_fn(seq_model: str,
                      embed_config,
                      model_config) -> hk.TransformedWithState:
  """Creates a Haiku transform that also returns pre-classifier embeddings."""
  if seq_model != 'transformer':
    raise ValueError('This analysis currently supports only transformer models.')

  def _forward(examples, labels, mask, is_training):
    embedder = InputEmbedder(**embed_config)
    model = Transformer(embedder, **model_config)
    logits, embeddings = model(
        examples,
        labels,
        mask,
        is_training=is_training,
        return_embeddings=True)
    return logits, embeddings

  return hk.transform_with_state(_forward)


def _sample_fewshot_sequence(seq_generator,
                             rng: np.random.Generator,
                             shots: int,
                             ways: int,
                             randomly_generate_rare: bool,
                             grouped: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Samples a common few-shot sequence and returns examples, labels, classes."""
  seq_len = shots * ways + 1
  classes_to_sample, class_weights = seq_generator._get_classes_to_sample('common')  # pylint: disable=protected-access
  label_options = list(range(ways))
  class_options_idx = rng.choice(
      len(classes_to_sample),
      size=ways,
      replace=True,
      p=class_weights)
  class_options = [classes_to_sample[i] for i in class_options_idx]
  query_idx = rng.integers(0, ways)
  query_label = label_options[query_idx]
  query_class = class_options[query_idx]

  seq_labels = np.array(label_options * shots + [query_label], dtype=np.int32)
  seq_classes = np.array(class_options * shots + [query_class], dtype=np.int32)
  seq_examples = seq_generator._create_noisy_image_seq(  # pylint: disable=protected-access
      seq_classes,
      randomly_generate_rare=randomly_generate_rare)

  ordering = np.arange(seq_len - 1)
  if grouped:
    for i in range(shots):
      rng.shuffle(ordering[i * ways:(i + 1) * ways])
  else:
    rng.shuffle(ordering)
  seq_labels[:-1] = seq_labels[ordering]
  seq_examples[:-1] = seq_examples[ordering]
  seq_classes[:-1] = seq_classes[ordering]

  return seq_examples, seq_labels, seq_classes


def _sample_no_support_sequence(seq_generator,
                                rng: np.random.Generator,
                                seq_len: int,
                                labeling: str,
                                randomly_generate_rare: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Samples a common no-support sequence and returns examples, labels, classes."""
  classes_to_sample, class_weights = seq_generator._get_classes_to_sample('common')  # pylint: disable=protected-access
  query_class_idx = rng.choice(len(classes_to_sample), p=class_weights)
  remaining_idx = np.delete(np.arange(len(classes_to_sample)), query_class_idx)
  remaining_weights = np.delete(class_weights, query_class_idx)
  remaining_weights /= np.sum(remaining_weights)
  support_idx = rng.choice(
      remaining_idx,
      size=seq_len - 1,
      replace=True,
      p=remaining_weights)
  rng.shuffle(support_idx)
  seq_class_idx = np.concatenate([support_idx, [query_class_idx]])
  seq_classes = np.array([classes_to_sample[i] for i in seq_class_idx], dtype=np.int32)
  seq_examples = seq_generator._create_noisy_image_seq(  # pylint: disable=protected-access
      seq_classes,
      randomly_generate_rare=randomly_generate_rare)

  if labeling == 'original':
    seq_labels = seq_classes.copy()
  elif labeling == 'ordered':
    seq_labels = seq_class_idx.copy()
    seq_labels += seq_generator.n_rare_classes
  elif labeling.startswith('ordered'):
    seq_labels = seq_class_idx.copy()
    label_start = int(labeling.split('ordered')[1])
    seq_labels += label_start
  else:
    raise ValueError(f'Unsupported labeling scheme: {labeling}')

  seq_labels = seq_labels.astype(np.int32)
  return seq_examples, seq_labels, seq_classes


def _collect_condition_embeddings(condition_name: str,
                                  label_filter: Optional[int],
                                  sampler,
                                  params,
                                  state,
                                  forward_apply,
                                  hk_rng,
                                  samples_per_class: int,
                                  max_attempts: int,
                                  common_classes: Sequence[int]) -> List[Dict[str, object]]:
  """Collects embeddings for a single condition."""
  per_class_counts = collections.Counter()
  entries = []
  attempts = 0
  while attempts < max_attempts:
    attempts += 1
    seq_examples, seq_labels, seq_classes = sampler()
    query_class = int(seq_classes[-1])
    if query_class not in common_classes:
      continue
    if per_class_counts[query_class] >= samples_per_class:
      continue
    query_label = int(seq_labels[-1]) if seq_labels.size else None
    if label_filter is not None and query_label != label_filter:
      continue

    batch_examples = np.expand_dims(seq_examples, 0)
    batch_labels = np.expand_dims(seq_labels, 0)
    (logits, embeddings), _ = forward_apply(
        params,
        state,
        rng=next(hk_rng),
        examples=batch_examples,
        labels=batch_labels,
        mask=None,
        is_training=False)
    del logits
    representation = np.asarray(embeddings)[0, -1]

    entries.append({
        'representation': representation,
        'class_id': query_class,
        'context_label': label_filter,
        'condition': condition_name,
    })
    per_class_counts[query_class] += 1
    if all(per_class_counts[c] >= samples_per_class for c in common_classes):
      break

  missing = [c for c in common_classes if per_class_counts[c] < samples_per_class]
  if missing:
    logging.warning('%s: insufficient samples for classes: %s', condition_name, missing)
  else:
    logging.info('%s: collected %d embeddings', condition_name, len(entries))
  return entries


def _prepare_forward_and_state(cfg) -> Tuple[hk.Params, hk.State, hk.TransformedWithState, experiment_lib.Experiment]:
  """Instantiates the experiment and loads the checkpoint."""
  analysis_experiment = experiment_lib.Experiment(
      mode='eval_fewshot_common', init_rng=0, config=cfg)
  forward = _build_forward_fn(cfg.seq_model, analysis_experiment.embed_config,
                              analysis_experiment.model_config)
  params, state = _load_checkpoint(FLAGS.checkpoint)
  params = jax.device_put(params)
  state = jax.device_put(state)
  return params, state, forward, analysis_experiment


def _run_projection(data: List[Dict[str, object]],
                    method: str,
                    neighbors: int,
                    min_dist: float) -> Tuple[np.ndarray, np.ndarray]:
  """Projects the collected representations into 2-D."""
  representations = np.stack([entry['representation'] for entry in data])
  if method == 'umap':
    try:
      import umap  # pylint: disable=g-import-not-at-top
    except ImportError as exc:
      raise ImportError(
          'UMAP is not installed. Install umap-learn or use --projection_method=pca.'
      ) from exc
    reducer = umap.UMAP(n_components=2, n_neighbors=neighbors, min_dist=min_dist)
    coords = reducer.fit_transform(representations)
  elif method == 'pca':
    centered = representations - np.mean(representations, axis=0, keepdims=True)
    if centered.shape[0] == 1:
      coords = np.zeros((1, 2), dtype=centered.dtype)
    else:
      _, _, vh = np.linalg.svd(centered, full_matrices=False)
      components = vh[:2].T
      coords = centered @ components
      if coords.shape[1] < 2:
        pad_width = 2 - coords.shape[1]
        coords = np.pad(coords, ((0, 0), (0, pad_width)), mode='constant')
  else:
    raise ValueError(f'Unsupported projection method: {method}')
  return representations, coords

def _compute_pairwise_distances(representations: np.ndarray) -> np.ndarray:
  """Computes a dense pairwise Euclidean distance matrix."""
  sq_norms = np.sum(representations ** 2, axis=1, keepdims=True)
  dist_sq = sq_norms + sq_norms.T - 2 * representations @ representations.T
  dist_sq = np.maximum(dist_sq, 0.0)
  return np.sqrt(dist_sq)


def _silhouette_score(representations: np.ndarray, labels: np.ndarray) -> float:
  """Computes the mean silhouette score for the given representations."""
  if representations.shape[0] < 2:
    return np.nan
  unique_labels = np.unique(labels)
  if unique_labels.size < 2:
    return np.nan

  distances = _compute_pairwise_distances(representations)
  cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}
  silhouettes = np.zeros(representations.shape[0], dtype=np.float32)

  for idx in range(representations.shape[0]):
    label = labels[idx]
    same_cluster = cluster_indices[label]
    if same_cluster.size <= 1:
      silhouettes[idx] = 0.0
      continue
    a = (np.sum(distances[idx, same_cluster]) - 0.0) / (same_cluster.size - 1)
    b = np.inf
    for other_label, other_idx in cluster_indices.items():
      if other_label == label:
        continue
      if other_idx.size == 0:
        continue
      b = min(b, np.mean(distances[idx, other_idx]))
    if not np.isfinite(b):
      silhouettes[idx] = 0.0
      continue
    denom = max(a, b)
    silhouettes[idx] = 0.0 if denom == 0 else (b - a) / denom

  return float(np.mean(silhouettes))


def _compute_silhouette_metrics(
    data: List[Dict[str, object]],
    output_dir: str,
) -> None:
  """Computes silhouette metrics for no-support and context label comparisons."""
  representations = [entry.get('representation') for entry in data]
  if any(rep is None for rep in representations):
    logging.warning('Skipping silhouette analysis because representations are missing.')
    return

  os.makedirs(output_dir, exist_ok=True)
  metrics_path = os.path.join(output_dir, 'silhouette_scores.csv')

  no_support_entries = [entry for entry in data if entry['condition'] == 'no_support']
  no_support_score = np.nan
  if len(no_support_entries) >= 2:
    no_support_reps = np.stack([entry['representation'] for entry in no_support_entries])
    no_support_labels = np.array([entry['class_id'] for entry in no_support_entries])
    no_support_score = _silhouette_score(no_support_reps, no_support_labels)
  else:
    logging.warning('Not enough no-support samples to compute silhouette.')

  context_entries = [
      entry for entry in data
      if entry['condition'] in {'context_label_0', 'context_label_1'}
  ]
  class_ids = sorted({entry['class_id'] for entry in context_entries})
  per_class_scores = []
  per_class_rows = []
  for class_id in class_ids:
    class_entries = [
        entry for entry in context_entries if entry['class_id'] == class_id
    ]
    if len(class_entries) < 2:
      continue
    labels = np.array([entry['context_label'] for entry in class_entries])
    if np.unique(labels).size < 2:
      continue
    reps = np.stack([entry['representation'] for entry in class_entries])
    score = _silhouette_score(reps, labels)
    per_class_scores.append(score)
    per_class_rows.append({
        'metric': 'context_label_silhouette',
        'class_id': class_id,
        'value': score,
        'num_samples': len(class_entries),
        'num_clusters': np.unique(labels).size,
    })

  avg_context_score = float(np.mean(per_class_scores)) if per_class_scores else np.nan

  with open(metrics_path, 'w', newline='') as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=['metric', 'class_id', 'value', 'num_samples', 'num_clusters'])
    writer.writeheader()
    writer.writerow({
        'metric': 'no_support_silhouette',
        'class_id': '',
        'value': no_support_score,
        'num_samples': len(no_support_entries),
        'num_clusters': len({entry['class_id'] for entry in no_support_entries})
        if no_support_entries else 0,
    })
    writer.writerow({
        'metric': 'avg_context_label_silhouette',
        'class_id': '',
        'value': avg_context_score,
        'num_samples': len(context_entries),
        'num_clusters': 2,
    })
    writer.writerows(per_class_rows)

  logging.info('Silhouette (no_support / class labels): %s', no_support_score)
  logging.info('Silhouette (avg per-class context labels): %s', avg_context_score)
  logging.info('Saved silhouette metrics to %s', metrics_path)


def _save_metadata(output_dir: str,
                   coords: np.ndarray,
                   data: List[Dict[str, object]],
                   method: str) -> None:
  """Saves metadata and coordinates for inspection."""
  os.makedirs(output_dir, exist_ok=True)
  csv_path = os.path.join(output_dir, f'{method}_metadata.csv')
  with open(csv_path, 'w', newline='') as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=['x', 'y', 'class_id', 'condition', 'context_label'])
    writer.writeheader()
    for coord, entry in zip(coords, data):
      writer.writerow({
          'x': float(coord[0]),
          'y': float(coord[1]),
          'class_id': entry['class_id'],
          'condition': entry['condition'],
          'context_label': entry['context_label']
      })
  logging.info('Saved %s metadata to %s', method.upper(), csv_path)

  npz_path = os.path.join(output_dir, f'{method}_data.npz')
  np.savez(
      npz_path,
      coords=coords,
      class_ids=np.array([entry['class_id'] for entry in data]),
      conditions=np.array([entry['condition'] for entry in data]),
      context_labels=np.array([
          -1 if entry['context_label'] is None else entry['context_label']
          for entry in data
      ]),
      representations=np.stack([entry['representation'] for entry in data]),
      projection_method=method)
  logging.info('Saved %s raw arrays to %s', method.upper(), npz_path)


def _load_metadata_npz(npz_path: str) -> Tuple[np.ndarray, List[Dict[str, object]], str]:
  """Loads cached metadata and coordinates."""
  if not os.path.exists(npz_path):
    raise FileNotFoundError(f'Metadata file not found: {npz_path}')
  with np.load(npz_path, allow_pickle=True) as payload:
    coords = payload['coords']
    class_ids = payload['class_ids']
    conditions = payload['conditions']
    context_labels = payload['context_labels']
    representations = payload.get('representations')
    projection_method = payload.get('projection_method')
  method = None
  if projection_method is not None:
    method = str(np.array(projection_method).item())
  if not method:
    method = FLAGS.projection_method
    logging.warning('Metadata missing projection method; falling back to flag value %s',
                    method)

  entries: List[Dict[str, object]] = []
  for idx in range(coords.shape[0]):
    ctx_label = int(context_labels[idx])
    entries.append({
        'representation': None if representations is None else representations[idx],
        'class_id': int(class_ids[idx]),
        'condition': str(conditions[idx]),
        'context_label': None if ctx_label < 0 else ctx_label,
    })
  return coords, entries, method


import seaborn as sns

def _plot_projection(output_dir: str,
                            coords: np.ndarray,
                            data: List[Dict[str, object]],
                            common_classes: Optional[Sequence[int]],
                            method: str) -> None:
    """Minimalistic, seaborn-grayblue publication-style 2D projection visualization."""
    os.makedirs(output_dir, exist_ok=True)

    # Prepare color and marker mappings
    if common_classes is None:
        unique_classes = sorted({int(entry['class_id']) for entry in data})
    else:
        unique_classes = sorted(set(common_classes))

    # cmap = plt.cm.get_cmap('turbo', len(unique_classes))
    # class_to_color = {cls: cmap(i) for i, cls in enumerate(unique_classes)}
    palette = sns.color_palette("Paired", len(unique_classes ))

    class_to_color = {cls: palette[i] for i, cls in enumerate(unique_classes)}
    # class_to_color = {unique_classes[0]: palette[1], unique_classes[1]: palette[3], unique_classes[2]: palette[6]}

    condition_to_marker = {
        'context_label_0': 'o',
        'context_label_1': 's',
        'no_support': '^'
    }

    # === seaborn gray-blue background ===
    sns.set_style("whitegrid")   # ensures seaborn-like aesthetics
    seaborn_bg = "#E5EAF2"       # gray-blue used by seaborn themes

    plt.figure(figsize=(8, 6), facecolor=seaborn_bg)
    ax = plt.gca()
    ax.set_facecolor(seaborn_bg)

    # ax.set_xlim(0, 15)
    # # ax.set_ylim(0, 15)
    # ax.set_ylim(10, 25)
    # Remove axes and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # === Scatter plotting ===
    for condition, marker in condition_to_marker.items():
        indices = [i for i, entry in enumerate(data) if entry['condition'] == condition]
        if not indices:
            continue

        colors = [class_to_color[data[i]['class_id']] for i in indices]

        plt.scatter(
            coords[indices, 0],
            coords[indices, 1],
            c=colors,
            marker=marker,
            edgecolors='white',
            linewidths=0.3,
            s=75,
            alpha=0.95,
            label=condition
        )



    # === Legends (outside plot) ===
    from matplotlib.lines import Line2D

    class_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=class_to_color[cls],
               label=f'class_{cls}', markersize=6,
               markeredgecolor='white')
        for cls in unique_classes
    ]
    condition_handles = [
        Line2D([0], [0], marker=marker, color='k',
               linestyle='', label=label)
        for label, marker in condition_to_marker.items()
    ]

    # Legend 1: context label
    # first_legend = plt.legend(
    #     handles=condition_handles,
    #     title='Context label',
    #     bbox_to_anchor=(1.03, 1.0),
    #     loc='upper left',
    #     frameon=False,
    #     fontsize=9
    # )
    # ax.add_artist(first_legend)
    #
    # Legend 2: class colors
    # plt.legend(
    #     handles=class_handles,
    #     title='Common class',
    #     # bbox_to_anchor=(1.03, 0.35),
    #     loc='upper left',
    #     frameon=False,
    #     fontsize=8
    # )

    plt.tight_layout(rect=[0, 0, 0.8, 1])  # leave room for legends

    fig_path = os.path.join(output_dir, f'common_context_{method}.png')
    plt.savefig(fig_path, dpi=400, bbox_inches='tight')
    plt.close()
    logging.info('Saved seaborn-polished %s figure to %s', method.upper(), fig_path)



def main(argv: Sequence[str]) -> None:
  del argv

  if FLAGS.metadata_npz:
    coords, data_entries, method = _load_metadata_npz(FLAGS.metadata_npz)
    logging.info('Loaded %d cached samples from %s', len(data_entries),
                 FLAGS.metadata_npz)
    _compute_silhouette_metrics(data_entries, FLAGS.output_dir)
    _plot_projection(FLAGS.output_dir, coords, data_entries, None, method)
    return

  if not FLAGS.analysis_config or not FLAGS.checkpoint:
    raise ValueError(
        '--analysis_config and --checkpoint must be provided unless --metadata_npz '
        'is set.')

  cfg = _load_training_config(FLAGS.analysis_config)
  seq_cfg = cfg.data.seq_config
  shots = int(seq_cfg.fs_shots)
  ways = int(seq_cfg.ways)
  seq_len_no_support = int(seq_cfg.seq_len)
  labeling_common = seq_cfg.labeling_common
  randomly_generate_rare = bool(seq_cfg.randomly_generate_rare)
  grouped = bool(seq_cfg.grouped)

  params, state, forward, analysis_experiment = _prepare_forward_and_state(cfg)
  seq_generator = analysis_experiment.data_generator_factory
  forward_apply = forward.apply
  hk_rng = hk.PRNGSequence(FLAGS.model_seed)
  numpy_rng = np.random.default_rng(FLAGS.sampling_seed)
  common_classes = list(seq_generator.common_classes)
  # common_classes = [1316, 580, 243]

  def no_support_sampler():
    return _sample_no_support_sequence(seq_generator, numpy_rng, seq_len_no_support,
                                       labeling_common, randomly_generate_rare)

  logging.info('Collecting embeddings...')
  data_entries: List[Dict[str, object]] = []
  data_entries.extend(
      _collect_condition_embeddings(
          'context_label_0',
          label_filter=0,
          sampler=lambda: _sample_fewshot_sequence(
              seq_generator, numpy_rng, shots, ways, randomly_generate_rare,
              grouped),
          params=params,
          state=state,
          forward_apply=forward_apply,
          hk_rng=hk_rng,
          samples_per_class=FLAGS.samples_per_class,
          max_attempts=FLAGS.max_sampling_attempts,
          common_classes=common_classes))
  data_entries.extend(
      _collect_condition_embeddings(
          'context_label_1',
          label_filter=1,
          sampler=lambda: _sample_fewshot_sequence(
              seq_generator, numpy_rng, shots, ways, randomly_generate_rare,
              grouped),
          params=params,
          state=state,
          forward_apply=forward_apply,
          hk_rng=hk_rng,
          samples_per_class=FLAGS.samples_per_class,
          max_attempts=FLAGS.max_sampling_attempts,
          common_classes=common_classes))
  data_entries.extend(
      _collect_condition_embeddings(
          'no_support',
          label_filter=None,
          sampler=no_support_sampler,
          params=params,
          state=state,
          forward_apply=forward_apply,
          hk_rng=hk_rng,
          samples_per_class=FLAGS.samples_per_class,
          max_attempts=FLAGS.max_sampling_attempts,
          common_classes=common_classes))

  if not data_entries:
    raise RuntimeError('No embeddings were collected. Check sampling parameters.')

  method = FLAGS.projection_method
  _compute_silhouette_metrics(data_entries, FLAGS.output_dir)
  # _, coords = _run_projection(
  #     data_entries,
  #     method=method,
  #     neighbors=FLAGS.umap_neighbors,
  #     min_dist=FLAGS.umap_min_dist)
  # logging.info('Computed %s coordinates for %d samples.', method.upper(), len(coords))

  # _save_metadata(FLAGS.output_dir, coords, data_entries, method)
  # # _plot_projection(FLAGS.output_dir, coords, data_entries, common_classes, method)
  # _plot_projection(FLAGS.output_dir, coords, data_entries, common_classes, method)

if __name__ == '__main__':
  app.run(main)