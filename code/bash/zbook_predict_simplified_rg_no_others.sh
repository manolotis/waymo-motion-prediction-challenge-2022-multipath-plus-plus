BATCH_SIZE=16
N_JOBS=12
BASE_SCRIPT="/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/code/predict.py"
OUT_PATH="/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/predictions_simplified_rg_no_others/"

python $BASE_SCRIPT \
  --config /home/manolotis/sandbox/temporal-consistency-tests/multipathPP/code/configs/predict.yaml \
  --test-data-path "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/data/prerendered/validation_simplified_rg_no_others/" \
  --batch-size $BATCH_SIZE \
  --n-jobs $N_JOBS \
  --n-shards 1 \
  --out-path $OUT_PATH \
  --model-name "final_RoP_Cov_Single__18c3cff"

