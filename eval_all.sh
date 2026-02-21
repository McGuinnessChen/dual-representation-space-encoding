#!/bin/bash

BASE_DIR="./dual_transformer_omniglot_38_layer4_mu5_small_p0.9_z2/models/latest"

for CKP_DIR in $BASE_DIR/*; do

  if [ -f stop_run ]; then
    echo "Detected stop_run file. Exiting loop."
    break
  fi

  if [ -d "$CKP_DIR" ]; then
    echo "Running on checkpoint: $CKP_DIR"
    python -m dual-representation-space-encoding.experiment.experiment \
      --config dual-representation-space-encoding/experiment/configs/images_all_exemplars.py \
      --logtostderr \
      --config.one_off_evaluate \
      --config.restore_path="$CKP_DIR" \
      --jaxline_mode=eval_fewshot_holdout
  fi

  if [ -d "$CKP_DIR" ]; then
    echo "Running on checkpoint: $CKP_DIR"
    python -m dual-representation-space-encoding.experiment.experiment \
      --config dual-representation-space-encoding/experiment/configs/images_all_exemplars.py \
      --logtostderr \
      --config.one_off_evaluate \
      --config.restore_path="$CKP_DIR" \
      --jaxline_mode=eval_no_support_zipfian
  fi
done
