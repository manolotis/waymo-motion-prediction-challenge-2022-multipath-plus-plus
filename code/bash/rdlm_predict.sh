python /home/manolotis/sandbox/scenario_based_evaluation/multipathPP/code/predict.py \
  --config /home/manolotis/sandbox/scenario_based_evaluation/multipathPP/code/configs/predict.yaml \
  --test-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/testing" \
  --batch-size 64 \
  --n-jobs 32 \
  --n-shards 1 \
  --out-path "/home/manolotis/sandbox/scenario_based_evaluation/multipathPP/predictions/"\
  --model-name "final_RoP_Cov_Single__18c3cff"
