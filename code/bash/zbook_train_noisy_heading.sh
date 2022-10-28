python /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/train.py \
  --config /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/final_RoP_Cov_Single_lr4e-4_noisy_heading.yaml \
  --train-data-path "/home/manolotis/sandbox/robustness_benchmark/multipathPP/data/prerendered/training" \
  --val-data-path "/home/manolotis/sandbox/robustness_benchmark/multipathPP/data/prerendered/validation" \
  --train-data-path-noisy "/home/manolotis/sandbox/robustness_benchmark/multipathPP/data/prerendered/training_noisy_heading" \
  --val-data-path-noisy "/home/manolotis/sandbox/robustness_benchmark/multipathPP/data/prerendered/validation_noisy_heading" \
  --batch-size 4 \
  --n-jobs 8 \
  --n-shards 16