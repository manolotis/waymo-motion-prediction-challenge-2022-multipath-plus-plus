python /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/predict.py \
  --config /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/predict_no_road.yaml \
  --test-data-path "/home/manolotis/sandbox/robustness_benchmark/multipathPP/data/prerendered/test" \
  --batch-size 4 \
  --n-jobs 2 \
  --n-shards 1 \
  --out-path "/home/manolotis/sandbox/robustness_benchmark/multipathPP/predictions/"\
  --model-name "final_RoP_Cov_Single_lr4e-4__65c803f" \
  --model-name-addition "no_road"