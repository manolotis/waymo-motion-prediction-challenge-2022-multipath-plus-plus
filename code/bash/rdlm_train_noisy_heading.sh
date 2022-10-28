python /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/train.py \
  --config /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/final_RoP_Cov_Single_lr4e-4_noisy_heading.yaml \
  --train-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/training_noisy_heading" \
  --val-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/validation_noisy_heading" \
  --batch-size 84 \
  --n-shards 16