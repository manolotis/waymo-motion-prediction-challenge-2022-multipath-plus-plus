BATCH_SIZE=64
N_JOBS=32
BASE_SCRIPT="/home/manolotis/sandbox/robustness_benchmark/multipathPP/code/predict.py"
OUT_PATH="/home/manolotis/sandbox/robustness_benchmark/multipathPP/predictions/"

python $BASE_SCRIPT \
  --config /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/predict.yaml \
  --test-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/testing" \
  --batch-size $BATCH_SIZE \
  --n-jobs $N_JOBS \
  --n-shards 1 \
  --out-path $OUT_PATH \
  --model-name "final_RoP_Cov_Single__18c3cff"


python /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/predict.py \
  --config /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/predict_no_past.yaml \
  --test-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/testing" \
  --batch-size $BATCH_SIZE \
  --n-jobs $N_JOBS \
  --n-shards 1 \
  --out-path $OUT_PATH \
  --model-name "final_RoP_Cov_Single__18c3cff" \
  --model-name-addition "no_past"

python /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/predict.py \
  --config /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/predict_no_road.yaml \
  --test-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/testing" \
  --batch-size $BATCH_SIZE \
  --n-jobs $N_JOBS \
  --n-shards 1 \
  --out-path $OUT_PATH \
  --model-name "final_RoP_Cov_Single__18c3cff" \
  --model-name-addition "no_road"

python /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/predict.py \
  --config /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/predict_noisy_heading.yaml \
  --test-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/testing_noisy_heading" \
  --batch-size $BATCH_SIZE \
  --n-jobs $N_JOBS \
  --n-shards 1 \
  --out-path $OUT_PATH \
  --model-name "final_RoP_Cov_Single__18c3cff" \
  --model-name-addition "noisy_heading"

python /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/predict.py \
  --config /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/predict_noisy_heading.yaml \
  --test-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/testing_noisy_heading" \
  --batch-size $BATCH_SIZE \
  --n-jobs $N_JOBS \
  --n-shards 1 \
  --out-path $OUT_PATH \
  --model-name "final_RoP_Cov_Single_noisy_heading__18c3cff" \
  --model-name-addition "retrained"

python /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/predict.py \
  --config /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/predict.yaml \
  --test-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/testing" \
  --batch-size $BATCH_SIZE \
  --n-jobs $N_JOBS \
  --n-shards 1 \
  --out-path $OUT_PATH \
  --model-name "final_RoP_Cov_Single_noisy_heading__18c3cff" \
  --model-name-addition "retrained_unperturbed"

#python /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/predict.py \
#  --config /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/predict_no_past.yaml \
#  --test-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/testing" \
#  --batch-size $BATCH_SIZE \
#  --n-jobs $N_JOBS \
#  --n-shards 1 \
#  --out-path $OUT_PATH \
#  --model-name "final_RoP_Cov_Single_no_past__a4c65b3" \
#  --model-name-addition "retrained"
#
#python /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/predict.py \
#  --config /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/predict.yaml \
#  --test-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/testing" \
#  --batch-size $BATCH_SIZE \
#  --n-jobs $N_JOBS \
#  --n-shards 1 \
#  --out-path $OUT_PATH \
#  --model-name "final_RoP_Cov_Single_no_past__a4c65b3" \
#  --model-name-addition "retrained_unperturbed"

#python /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/predict.py \
#  --config /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/predict_no_road.yaml \
#  --test-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/testing" \
#  --batch-size $BATCH_SIZE \
#  --n-jobs $N_JOBS \
#  --n-shards 1 \
#  --out-path $OUT_PATH \
#  --model-name "final_RoP_Cov_Single_no_road__a4c65b3" \
#  --model-name-addition "retrained"
#
#python /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/predict.py \
#  --config /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/predict.yaml \
#  --test-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/testing" \
#  --batch-size $BATCH_SIZE \
#  --n-jobs $N_JOBS \
#  --n-shards 1 \
#  --out-path $OUT_PATH \
#  --model-name "final_RoP_Cov_Single_no_road__a4c65b3" \
#  --model-name-addition "retrained_unperturbed"