CUDA_VISIBLE_DEVICES=-1 \
python /home/manolotis/sandbox/scenario_based_evaluation/multipathPP/code/prerender/prerender.py \
  --data-path "/home/manolotis/sandbox/datasets/waymo_v1.1/uncompressed/tf_example/testing/" \
  --output-path "/home/manolotis/sandbox/scenario_based_evaluation/multipathPP/data/prerendered/testing/" \
  --config "/home/manolotis/sandbox/scenario_based_evaluation/multipathPP/code/configs/prerender.yaml" \
  --n-jobs 8