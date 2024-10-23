BATCH_SIZE=6
N_JOBS=8
BASE_SCRIPT="/home/manolotis/sandbox/multipathpp/code/predict.py"
OUT_PATH="/home/manolotis/sandbox/multipathpp/predictions_retrained_train/"
#OUT_PATH="/home/manolotis/sandbox/multipathpp/predictions/"

python $BASE_SCRIPT \
  --config /home/manolotis/sandbox/multipathpp/code/configs/predict_retrained.yaml \
  --test-data-path "/home/manolotis/sandbox/multipathpp/data/prerendered/train/" \
  --batch-size $BATCH_SIZE \
  --n-jobs $N_JOBS \
  --n-shards 1 \
  --out-path $OUT_PATH \
  --model-name "final_RoP_Cov_Single__aa8678f__from_trained"
#  --model-name "final_RoP_Cov_Single__aa8678f"
#  --model-name "final_RoP_Cov_Single__18c3cff"