<div align="center">

<h1>Reconciling In-Context and In-Weight Learning via Dual Representation Space Encoding</h1>

Guanyu Chen<sup>1</sup>, Ruichen Wang<sup>1</sup>, [Tianren Zhang](https://trzhang0116.github.io/)<sup>1†</sup>, Feng Chen<sup>1†</sup>


<sup>1</sup> Department of Automation, Tsinghua University
<br/>
<sup>†</sup> Co-corresponding authors



[//]: # (<a href="https://openreview.net/forum?id=bJK7VIOWAU">)

[//]: # (  <img src="https://img.shields.io/badge/OpenReview-bJK7VIOWAU-blue" alt="OpenReview">)

[//]: # (</a>)

[//]: # (<a href="https://openreview.net/pdf?id=bJK7VIOWAU">)

[//]: # (  <img src="https://img.shields.io/badge/Paper-PDF-red" alt="Paper PDF">)

[//]: # (</a>)

[//]: # (<img src="https://img.shields.io/badge/TMLR-02%2F2026-4b8bbe" alt="TMLR">)

</div>

## Overview

Transformers often exhibit a tension between **in-context learning (ICL)** and **in-weight learning (IWL)**.  
This repo implements **CoQE**, a simple architectural modification that **separately encodes context and query samples into two distinct representation spaces**, and composes them via an inner product, which empirically reconciles ICL and IWL across synthetic settings.

**Key idea.** Standard Transformers encode contextual/task information and sample information in a shared space, which can cause interference. CoQE decouples them into:

- **Task representation space** (contextual encoder)
- **Sample representation space** (token-wise/sample encoder)

The experiments involve training and evaluating a transformer on sequences of
[Omniglot](https://github.com/brendenlake/omniglot) image-label pairs, to elicit
and measure (few-shot) in-context learning vs in-weights learning. See Sec 2 of
the paper for an overview of the experimental design.


## Installation

To install the necessary requirements:

```shell
python3 -m venv eicl_venv
source eicl_venv/bin/activate
pip install --upgrade pip
pip install -r ./emergent_in_context_learning/requirements.txt
```

## Usage

### Default configs

Default experiment configurations are provided in `configs/`, and can be used
in `$PATH_TO_CONFIG` in the launch commands below.

*   `images_all_exemplars.py`: Each character class consists of 20 image
    examples (the original Omniglot problem).
*   `images_augmented.py`: We augment the total number of classes to 8x the
    original number, by applying transformations to each image class: flip left
    or right + rotate 0, 90, 180, or 270 degrees.
*   `images_identical.py`: Each character class consists only of a single image
    (the 1st of the 20 examples provided in the original Omniglot dataset)
*   `symbolic.py`: (relatively untested; not used in the paper)

Config files can be edited or forked as desired.

### Varieties of data sequences + Configurations for each

Omniglot sequences are generated in `datasets/data_generators.py`.

The image classes are divided into training and holdout. Training classes can be
"common" or "rare". The training classes can be uniformly or Zipf-distributed
(jointly over both common and rare classes). Related configurations are set in
`config.data.generator_config`.

There are few different types of data sequences:

*   `bursty` : These are the canonical bursty (and non-bursty) sequences used in
    training in the paper
*   `no_support_common`, `no_support_rare`, `non_support_zipfian` : These
    sequences enforce that the query class does not appear anywhere in the
    context, and are the sequences used for evaluating in-weights learning in
    the paper. They can consist entirely of common classes, rare classes, or be
    Zipf-distributed over all training classes.
*   `fewshot_common`, `fewshot_rare`, `fewshot_zipfian`, `fewshot_holdout` :
    These sequence are standard k-shot n-way fewshot sequences, and are used for
    evaluating in-context learning in the paper. They can exist of holdout
    classes, common classes, rare classes, or be Zipf-distributed over all
    training classes.
*   `mixed`: A mix of standard fewshot and iid randomly generated sequences.

Sequence types are specified in `config.data.train_seqs` and in
`config.eval_modes` (with an additional `eval_` prefix). You may specify a list
of eval modes, to evaluate the same learner on multiple sequence types.

See `experiment/experiment.py: _get_ds_seqs` and `datasets/data_generators.py:
SeqGenerator` for more details on settings, which are specified in
`config.data.seq_config`.


### Launch commands

These commands should be executed from the directory that you cloned the
repository into.

To run training:

```shell
$ python -m emergent_in_context_learning.experiment.experiment --config $PATH_TO_CONFIG --jaxline_mode train --logtostderr
# (save checkpoints using Ctrl+C)
```

To evaluate a trained model, override `config.restore_path` with the
subdirectory of `config.checkpoint_dir` containing the relevant checkpoint
(`$CKPT_DIR` below).

To evaluate on in-context learning (on holdout classes):

```shell
$ python -m emergent_in_context_learning.experiment.experiment --config $PATH_TO_CONFIG --logtostderr --config.one_off_evaluate --config.restore_path $CKPT_DIR --jaxline_mode eval_fewshot_holdout
```

To evaluate on in-weights learning (on trained classes):

```shell
$ python -m emergent_in_context_learning.experiment.experiment --config $PATH_TO_CONFIG --logtostderr --config.one_off_evaluate --config.restore_path $CKPT_DIR --jaxline_mode eval_no_support_zipfian
```

### End-to-end LLaMA3 embedding workflow

You only need to download the LLaMA3 checkpoint; the repository provides the
clustering and exemplar selection steps required to feed the model embeddings
into the training pipeline.

1. **Download the LLaMA3 weights** from Hugging Face to a local directory. The
   remaining steps operate purely on the model files.
2. **Cluster token embeddings and pick exemplars** using the provided script:

   ```shell
   python -m emergent_in_context_learning.scripts.build_llama3_embeddings \
     --model /path/to/llama3/checkpoint \
     --output /path/to/llama3_embeddings.h5 \
     --num-classes 3200 \
     --exemplars-per-class 5 \
     --seed 0 \
     --skip-special-tokens
   ```

   The script extracts the model's input embedding table, keeps only tokens
   composed solely of English letters/underscores, projects them to 1024
   dimensions with PCA, L2-normalizes the projected vectors, and performs
   spherical k-means clustering with FAISS. It over-clusters by 1.5× the
   requested class count (e.g., 4800 for a 3200-class run or 2400 for a 1600-
   class run), keeps the largest clusters that contain more than the exemplar
   threshold (e.g., >5 or >10 tokens), then selects exemplars from those
   clusters. The saved embeddings, token IDs, and centroids live at
   `<seed>/<exemplars>/feat` inside the HDF5 file (matching the
   `llama3_embeddings.py` config) and have 1024-dimensional features after the
   PCA projection. Adjust `--num-classes`, `--exemplars-per-class`, or
   `--dataset-path` if you want a different split.
3. **Point the experiment config at your artifact.** Update the
   `DEFAULT_EMBED_PATH` and `DEFAULT_DATASET_PATH` values in
   `experiment/configs/llama3_embeddings.py` (or override them on the command
   line) so training consumes the generated file. `DEFAULT_EMBED_PATH` should
   point to the HDF5 file produced by `build_llama3_embeddings.py`, and
   `DEFAULT_DATASET_PATH` defaults to `0/5/feat`, matching the seed/exemplar
   layout inside that file (`<seed>/<exemplars>/feat`). The class constants in
   the config (`LLAMA3_TOTAL_CLASSES`, `LLAMA3_RARE_CLASSES`,
   `LLAMA3_COMMON_CLASSES`) mirror the 3200-class artifact: 1600 classes are
   treated as rare and 1600 as common during sequence generation. Then launch
   training with the commands above using `experiment/configs/llama3_embeddings.py`
   as the `$PATH_TO_CONFIG`. The LLaMA3 config defaults to `p_bursty=1.0` and
   `p_bursty_zipfian=0.0`, so sequences are always bursty and avoid zipfian
   sampling by default.

### Visualizing common-class query representations

The analysis script `analysis/common_context_umap.py` collects the transformer representation of the final (query) token for common classes across three evaluation conditions:

1. `eval_fewshot_common` sequences where the query class is labeled as `0` in the context.
2. `eval_fewshot_common` sequences where the query class is labeled as `1` in the context.
3. `eval_no_support_common` sequences where the query class does not appear in the context.

For each common class, the script gathers a configurable number of samples for each condition, projects the resulting representations to 2-D using UMAP (default) or PCA, and saves both the scatter plot and the underlying metadata (CSV + NPZ).

```shell
python -m emergent_in_context_learning.analysis.common_context_umap \
  --analysis_config experiment/configs/images_all_exemplars.py \
  --checkpoint $CKPT_DIR/checkpoint.dill \
  --output_dir /tmp/common_context_umap \
  --samples_per_class 32 \
  --projection_method umap  # or "pca"
```

`--analysis_config` should reference the same config that was used to train the
checkpoint (e.g., one of the files under `experiment/configs/`). You can supply
either a filesystem path such as
`experiment/configs/images_all_exemplars.py` or the corresponding Python module
path (`emergent_in_context_learning.experiment.configs.images_all_exemplars`).
Adjust `--checkpoint` to point to the directory or file containing
`checkpoint.dill`. Use `--projection_method pca` if you prefer PCA or you do not
have `umap-learn` installed; otherwise, keep the default UMAP setting and adjust
`--umap_neighbors` / `--umap_min_dist` as needed. The output directory will contain:

* `common_context_<method>.png`: Scatter plot colored by class and marked by context condition (`method` is `umap` or `pca`).
* `<method>_metadata.csv`: Coordinates and labels for each sample.
* `<method>_data.npz`: Raw representations, coordinates, class IDs, context labels, and the projection method.

If you already have a metadata NPZ file, you can regenerate the scatter plot without
loading a checkpoint:

```shell
python -m emergent_in_context_learning.analysis.common_context_umap \
  --metadata_npz /tmp/common_context_umap/umap_data.npz \
  --output_dir /tmp/common_context_umap
```

The script automatically infers the projection method from the NPZ (falling back to
`--projection_method` if necessary) and reuses the cached coordinates to produce a
fresh plot.

## Citing this work

If you use this work, please cite the following paper
```
@misc{chan_data_2022,
  title = {Data Distributional Properties Drive Emergent In-Context Learning in Transformers},
  author = {Chan, Stephanie C. Y. and Santoro, Adam and Lampinen, Andrew K. and Wang, Jane X. and Singh, Aaditya and Richemond, Pierre H. and McClelland, Jay and Hill, Felix},
  journal = {Neural Information Processing Systems},
  year = {2022},
}
```

We would also like to thank the following colleagues for their contributions to
the implementation of the transformer model:
Igor Babuschkin, Junyoung Chung, David Choi, Tamara Norman, Sebastian Borgeaud,
Jack Rae, David Saxton, Yujia Li, Phil Blunsom, Maribeth Rauh, Roman Ring,
Nate Kushman, Vinicius Zambaldi, Tom Hennigan


## License and disclaimer

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.