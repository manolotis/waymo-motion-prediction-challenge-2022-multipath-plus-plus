python /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/predict.py \
  --config /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/predict_noisy_heading.yaml \
  --test-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/testing_noisy_heading" \
  --batch-size 64 \
  --n-jobs 32 \
  --n-shards 1 \
  --out-path "/home/manolotis/sandbox/robustness_benchmark/multipathPP/predictions/"\
  --model-name "final_RoP_Cov_Single__18c3cff" \
  --model-name-addition "noisy_heading_retrained"