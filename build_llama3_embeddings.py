# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

"""Utility to cluster LLaMA3 token embeddings into exemplar classes."""

import argparse
import pathlib
import re
from typing import Iterable, Sequence, Tuple

import faiss
import h5py
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          'Cluster LLaMA3 token embeddings into synthetic classes and export '
          'them as an HDF5 file consumable by the training pipeline.'
      ))
  parser.add_argument(
      '--model',
      required=True,
      help='Hugging Face model name or local path for the LLaMA3 checkpoint.',
  )
  parser.add_argument(
      '--output',
      required=True,
      help='Destination HDF5 path for the clustered embeddings.',
  )
  parser.add_argument(
      '--num-classes',
      type=int,
      default=3200,
      help='Number of clusters/classes to create from the token embeddings.',
  )
  parser.add_argument(
      '--exemplars-per-class',
      type=int,
      default=5,
      help='Number of exemplar embeddings to keep from each cluster.',
  )
  parser.add_argument(
      '--seed',
      type=int,
      default=0,
      help='Random seed used for clustering and exemplar selection.',
  )
  parser.add_argument(
      '--dataset-path',
      default=None,
      help=(
          'Dataset path inside the HDF5 file. Defaults to "<seed>/<exemplars>/feat" '
          'to match the experiment config.'
      ),
  )
  parser.add_argument(
      '--skip-special-tokens',
      action='store_true',
      help='Drop special tokens (BOS/EOS/PAD/etc.) before clustering.',
  )
  parser.add_argument(
      '--extra-skip-ids',
      type=int,
      nargs='*',
      default=(),
      help='Additional token IDs to drop before clustering.',
  )
  parser.add_argument(
      '--dtype',
      default='float32',
      choices=['float32', 'float16'],
      help='Precision to use for embeddings prior to clustering.',
  )
  parser.add_argument(
      '--device',
      default=None,
      help='Optional device override (e.g., cuda:0); defaults to CPU.',
  )
  return parser.parse_args()


def _load_embeddings(model_id: str, dtype: torch.dtype, device: str | None) -> np.ndarray:
  model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
  if device:
    model = model.to(device)
  with torch.no_grad():
    embedding_weight = model.get_input_embeddings().weight.detach()
    embedding_weight = embedding_weight.to('cpu')
  return embedding_weight.numpy().astype(np.float32)


def _get_skip_ids(skip_special_tokens: bool, tokenizer: AutoTokenizer,
                  extra_skip_ids: Iterable[int]) -> Sequence[int]:
  skip_ids = []
  if skip_special_tokens:
    skip_ids.extend(tokenizer.all_special_ids)
  skip_ids.extend(extra_skip_ids)
  return skip_ids


def _is_english_token(token: str) -> bool:
  """Returns True if the token is composed only of ASCII letters or underscores."""

  if token is None:
    return False
  return bool(re.fullmatch(r'[A-Za-z_]+', token))


def _filter_embeddings(
    embeddings: np.ndarray,
    tokenizer: AutoTokenizer,
    skip_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, int, int]:
  """Filters out non-English tokens and any explicitly skipped IDs."""

  token_ids = np.arange(len(embeddings))
  mask = np.ones(len(embeddings), dtype=bool)

  # Keep only English tokens first to avoid clustering other scripts.
  tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
  english_mask = np.array([_is_english_token(tok) for tok in tokens], dtype=bool)
  mask &= english_mask
  num_english = int(english_mask.sum())

  if skip_ids:
    skip_ids_array = np.array(skip_ids, dtype=int)
    skip_ids_array = skip_ids_array[(skip_ids_array >= 0) & (skip_ids_array < len(mask))]
    mask[skip_ids_array] = False

  filtered = embeddings[mask]
  kept_token_ids = token_ids[mask]

  skipped_total = len(embeddings) - len(filtered)
  skipped_non_english = len(embeddings) - num_english

  return filtered, kept_token_ids, skipped_total, skipped_non_english


def _cluster_embeddings(
    embeddings: np.ndarray,
    num_classes: int,
    seed: int,
    *,
    warmup_niter: int = 5,   # 现在未使用，但保留签名
    final_niter: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if num_classes > len(embeddings):
        raise ValueError(
            f'num_classes ({num_classes}) must be <= number of available tokens ({len(embeddings)})')

    dim = embeddings.shape[1]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    embeddings = embeddings / norms
    filtered_embeddings = embeddings
    if len(filtered_embeddings) < num_classes:
        raise ValueError(
            f'not enough points ({len(filtered_embeddings)}) to form {num_classes} clusters.')

    print(f'Proceeding with spherical k-means on {len(filtered_embeddings)} tokens (dim={dim}).')

    final_clustering = faiss.Clustering(dim, num_classes)
    final_clustering.seed = seed
    final_clustering.niter = final_niter
    final_clustering.max_points_per_centroid = 100000
    final_clustering.min_points_per_centroid = 1
    final_clustering.verbose = True
    final_clustering.spherical = True                  # 关键：球面 k-means

    final_index = faiss.IndexFlatIP(dim)               # 关键：内积索引
    final_clustering.train(filtered_embeddings, final_index)

    # 用同一个内积索引做分配（越大越近）
    _, assignments = final_index.search(filtered_embeddings, 1)
    assignments = assignments[:, 0]

    # 取质心并单位化（稳妥起见再次归一化）
    centroids = faiss.vector_to_array(final_clustering.centroids).reshape(num_classes, dim)
    centroids_norm = np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-12)
    centroids = centroids / centroids_norm

    # ===== 4) 打印前 10 大簇大小 =====
    sizes = np.bincount(assignments, minlength=num_classes)
    top_order = np.argsort(sizes)[::-1]
    top_k = min(10, num_classes)
    print("Top-10 cluster sizes:")
    for rank, cid in enumerate(top_order[:top_k], start=1):
        print(f"#{rank}: cluster {cid} -> {sizes[cid]} points")

    # 与原签名兼容：不再过滤，返回全 True 的 mask
    top_two_mask = np.ones(len(embeddings), dtype=bool)
    return assignments, centroids, top_two_mask


def _overcluster_and_filter(
    embeddings: np.ndarray,
    num_classes: int,
    exemplars_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Over-clusters embeddings then keeps only sufficiently large clusters."""

  precluster_classes = 2800
  threshold = exemplars_per_class

  print(
      f'Over-clustering into {precluster_classes} classes, then keeping {num_classes} '
      f'clusters with more than {threshold} members')
  assignments_pre, centroids_pre, warmup_mask = _cluster_embeddings(
      embeddings, num_classes=precluster_classes, seed=seed)

  filtered_embeddings = embeddings[warmup_mask]
  cluster_sizes = np.bincount(assignments_pre, minlength=precluster_classes)
  eligible = np.flatnonzero(cluster_sizes > threshold)
  if len(eligible) < num_classes:
    raise ValueError(
        'Not enough clusters exceed the exemplar threshold: '
        f'found {len(eligible)} eligible clusters (> {threshold}) but need {num_classes}.')

  # Prefer the largest clusters to maximize available diversity per class.
  eligible_sorted = eligible[np.argsort(cluster_sizes[eligible])[::-1]]
  selected_clusters = eligible_sorted[:num_classes]

  selected_mask = np.isin(assignments_pre, selected_clusters)
  filtered_embeddings = filtered_embeddings[selected_mask]
  warmup_indices = np.flatnonzero(warmup_mask)
  filtered_assignment_indices = warmup_indices[selected_mask]
  filtered_assignments = assignments_pre[selected_mask]

  cluster_id_map = {old: new for new, old in enumerate(selected_clusters.tolist())}
  remapped_assignments = np.array(
      [cluster_id_map[idx] for idx in filtered_assignments], dtype=np.int64)

  selected_centroids = centroids_pre[selected_clusters]

  final_selected_mask = np.zeros(len(embeddings), dtype=bool)
  final_selected_mask[filtered_assignment_indices] = True

  print(
      f'Selected clusters cover {len(filtered_embeddings)} tokens; '
      f'min cluster size among kept: {cluster_sizes[selected_clusters].min()}')

  return remapped_assignments, selected_centroids, final_selected_mask


def _select_exemplars(embeddings: np.ndarray,
                      assignments: np.ndarray,
                      centroids: np.ndarray,
                      token_ids: np.ndarray,
                      exemplars_per_class: int,
                      seed: int) -> Tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  num_classes, dim = centroids.shape
  features = np.zeros((num_classes, exemplars_per_class, dim), dtype=np.float32)
  exemplar_token_ids = np.zeros((num_classes, exemplars_per_class), dtype=np.int64)

  for cluster_id in range(num_classes):
    mask = assignments == cluster_id
    cluster_embeddings = embeddings[mask]
    cluster_token_ids = token_ids[mask]
    if len(cluster_embeddings) == 0:
      raise ValueError(
          f'Cluster {cluster_id} is empty; reduce num_classes or adjust skip filters.')

    distances = np.sum((cluster_embeddings - centroids[cluster_id]) ** 2, axis=1)
    ordering = np.argsort(distances)[::-1]  # farthest first to encourage diversity
    chosen = ordering[:min(exemplars_per_class, len(ordering))].tolist()
    while len(chosen) < exemplars_per_class:
      chosen.append(int(rng.choice(ordering)))

    features[cluster_id] = cluster_embeddings[chosen]
    exemplar_token_ids[cluster_id] = cluster_token_ids[chosen]

  return features, exemplar_token_ids


def _save_embeddings(
    output: pathlib.Path,
    dataset_path: str,
    features: np.ndarray,
    token_ids: np.ndarray,
    centroids: np.ndarray,
    meta: dict[str, str | int],
) -> None:
  output.parent.mkdir(parents=True, exist_ok=True)
  with h5py.File(output, 'a') as f:
    if dataset_path in f:
      del f[dataset_path]
    f.create_dataset(dataset_path, data=features, compression='gzip')

    token_path = dataset_path.replace('/feat', '/token_ids')
    if token_path in f:
      del f[token_path]
    f.create_dataset(token_path, data=token_ids, compression='gzip')

    centroid_path = dataset_path.replace('/feat', '/centroids')
    if centroid_path in f:
      del f[centroid_path]
    f.create_dataset(centroid_path, data=centroids, compression='gzip')

    for key, value in meta.items():
      f.attrs[key] = value


def _dump_cluster_tokens_txt(
    tokenizer: AutoTokenizer,
    assignments: np.ndarray,
    token_ids: np.ndarray,
    txt_path: pathlib.Path,
) -> None:
  """将每个聚类对应的 token 词导出为 TXT。
  每行格式: <cluster_id>\t<count>\t<tok1> <tok2> ...
  """
  if len(assignments) != len(token_ids):
    raise ValueError("assignments 与 token_ids 长度不一致")

  num_classes = int(assignments.max()) + 1 if len(assignments) else 0
  buckets = [[] for _ in range(num_classes)]
  for tid, cid in zip(token_ids.tolist(), assignments.tolist()):
    buckets[cid].append(tid)

  txt_path.parent.mkdir(parents=True, exist_ok=True)
  with open(txt_path, "w", encoding="utf-8") as wf:
    for cid, tids in enumerate(buckets):
      toks = tokenizer.convert_ids_to_tokens(tids)
      wf.write(f"{cid}\t{len(toks)}\t{' '.join(toks)}\n")

  print(f"Wrote cluster->tokens mapping to: {txt_path}")



if __name__ == '__main__':
    args = _parse_args()
    dataset_path = args.dataset_path
    if not dataset_path:
        dataset_path = f'{args.seed}/{args.exemplars_per_class}/feat'

    torch_dtype = torch.float32 if args.dtype == 'float32' else torch.float16
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    skip_ids = _get_skip_ids(args.skip_special_tokens, tokenizer, args.extra_skip_ids)
    print('Loading embeddings...')
    raw_embeddings = _load_embeddings(args.model, torch_dtype, args.device)
    print(f'Loaded embeddings with shape {raw_embeddings.shape} (dim={raw_embeddings.shape[1]})')
    print('Filtering embeddings to tokens made of English letters/underscores...')
    embeddings, kept_token_ids, skipped_total, skipped_non_english = _filter_embeddings(
        raw_embeddings, tokenizer, skip_ids)
    print(
        f'Identified {len(raw_embeddings) - skipped_non_english} tokens composed only '
        f'of English letters/underscores (skipped {skipped_non_english} others)')
    print(f'Kept {len(embeddings)} tokens after filtering (skipped {skipped_total})')
    print('Clustering embeddings...')
    assignments, centroids, selected_mask = _overcluster_and_filter(
        embeddings=embeddings,
        num_classes=args.num_classes,
        exemplars_per_class=args.exemplars_per_class,
        seed=args.seed,
    )
    filtered_embeddings = embeddings[selected_mask]
    filtered_token_ids = kept_token_ids[selected_mask]
    print('Selecting exemplars...')
    features, exemplar_token_ids = _select_exemplars(
        embeddings=filtered_embeddings,
        assignments=assignments,
        centroids=centroids,
        token_ids=filtered_token_ids,
        exemplars_per_class=args.exemplars_per_class,
        seed=args.seed,
    )

    clusters_txt = pathlib.Path(args.output).with_suffix(".clusters.txt")
    _dump_cluster_tokens_txt(
        tokenizer=tokenizer,
        assignments=assignments,
        token_ids=filtered_token_ids,
        txt_path=clusters_txt,
    )
    print('Saving artifacts...')
    _save_embeddings(
        output=pathlib.Path(args.output),
        dataset_path=dataset_path,
        features=features,
        token_ids=exemplar_token_ids,
        centroids=centroids,
        meta={
            'model': args.model,
            'num_classes': args.num_classes,
            'exemplars_per_class': args.exemplars_per_class,
            'seed': args.seed,
            'english_only': True,
            'skip_special_tokens': args.skip_special_tokens,
            'extra_skip_ids': ','.join(map(str, args.extra_skip_ids)) if args.extra_skip_ids else '',
        },
    )
    print('Done.')