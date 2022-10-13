python /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/train.py \
  --config /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/final_RoP_Cov_Single_lr1e-2.yaml \
  --train-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/training" \
  --val-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/validation" \
  --batch-size 84 \
  --n-shards 16