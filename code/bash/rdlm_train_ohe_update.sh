python /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/train.py \
  --config /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/ohe_update.yaml \
  --train-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/training" \
  --val-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/validation" \
  --batch-size 64 \
  --n-shards 16