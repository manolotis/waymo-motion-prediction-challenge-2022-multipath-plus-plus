BATCH_SIZE=1
N_JOBS=12
BASE_SCRIPT="/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/code/predict.py"
OUT_PATH="/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/predictions_carla"

python $BASE_SCRIPT \
  --config /home/manolotis/sandbox/temporal-consistency-tests/multipathPP/code/configs/predict.yaml \
  --test-data-path "/home/manolotis/sandbox/c4r/data/last" \
  --batch-size $BATCH_SIZE \
  --n-jobs $N_JOBS \
  --n-shards 1 \
  --out-path $OUT_PATH \
  --model-name "final_RoP_Cov_Single__18c3cff"



