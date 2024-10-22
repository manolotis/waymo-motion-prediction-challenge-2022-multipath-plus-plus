BATCH_SIZE=64
N_JOBS=10
BASE_SCRIPT="/home/manolotis/sandbox/multipathpp/code/balance_data.py"
OUT_PATH="/home/manolotis/sandbox/multipathpp/predictions_retrained_train_dummy/" # will be ignored

python $BASE_SCRIPT \
  --config /home/manolotis/sandbox/multipathpp/code/configs/predict_retrained.yaml \
  --test-data-path "/home/manolotis/sandbox/multipathpp/data/prerendered/validation/" \
  --batch-size $BATCH_SIZE \
  --n-jobs $N_JOBS \
  --n-shards 1 \
  --out-path $OUT_PATH \
  --model-name "final_RoP_Cov_Single__aa8678f" \
#  --remove
#  --max-count 200000
#  --model-name "final_RoP_Cov_Single__18c3cff"