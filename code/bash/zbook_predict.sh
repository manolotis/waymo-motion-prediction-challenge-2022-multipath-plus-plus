BATCH_SIZE=8
N_JOBS=2
BASE_SCRIPT="/home/manolotis/sandbox/scenario_based_evaluation/multipathPP/code/predict.py"
OUT_PATH="/home/manolotis/sandbox/scenario_based_evaluation/multipathPP/predictions/"

python $BASE_SCRIPT \
  --config /home/manolotis/sandbox/scenario_based_evaluation/multipathPP/code/configs/predict.yaml \
  --test-data-path "/home/manolotis/sandbox/scenario_based_evaluation/multipathPP/data/prerendered/testing" \
  --batch-size $BATCH_SIZE \
  --n-jobs $N_JOBS \
  --n-shards 1 \
  --out-path $OUT_PATH \
  --model-name "final_RoP_Cov_Single__18c3cff"