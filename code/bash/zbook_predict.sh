BATCH_SIZE=8
N_JOBS=2
BASE_SCRIPT="/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/code/predict.py"
OUT_PATH="/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/predictions/"

python $BASE_SCRIPT \
  --config /home/manolotis/sandbox/temporal-consistency-tests/multipathPP/code/configs/predict.yaml \
  --test-data-path "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/data/prerendered/validation/" \
  --batch-size $BATCH_SIZE \
  --n-jobs $N_JOBS \
  --n-shards 1 \
  --out-path $OUT_PATH \
  --model-name "final_RoP_Cov_Single__18c3cff"