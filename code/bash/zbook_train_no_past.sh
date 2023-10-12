python /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/train.py \
  --config /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/final_RoP_Cov_Single_no_past.yaml \
  --train-data-path "/home/manolotis/sandbox/robustness_benchmark/multipathPP/data/prerendered/training" \
  --val-data-path "/home/manolotis/sandbox/robustness_benchmark/multipathPP/data/prerendered/validation" \
  --batch-size 8 \
  --n-jobs 2 \
  --n-shards 4