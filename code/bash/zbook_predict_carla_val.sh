BATCH_SIZE=8
N_JOBS=2
BASE_SCRIPT="/home/manolotis/sandbox/multipathpp/code/predict.py"
OUT_PATH="/home/manolotis/sandbox/multipathpp/predictions_validation/"
#OUT_PATH="/home/manolotis/sandbox/multipathpp/predictions_train/"
#OUT_PATH="/home/manolotis/sandbox/multipathpp/predictions/"

python $BASE_SCRIPT \
  --config /home/manolotis/sandbox/multipathpp/code/configs/predict.yaml \
  --test-data-path "/home/manolotis/sandbox/multipathpp/data/prerendered/validation/" \
  --batch-size $BATCH_SIZE \
  --n-jobs $N_JOBS \
  --n-shards 1 \
  --out-path $OUT_PATH \
  --model-name "final_RoP_Cov_Single__18c3cff"