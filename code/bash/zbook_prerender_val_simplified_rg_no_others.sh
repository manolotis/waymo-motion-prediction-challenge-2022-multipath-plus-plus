CUDA_VISIBLE_DEVICES=-1 \
python /home/manolotis/sandbox/temporal-consistency-tests/multipathPP/code/prerender/prerender.py \
  --data-path "/home/manolotis/sandbox/datasets/waymo_v1.1/uncompressed/tf_example/validation/" \
  --output-path "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/data/prerendered/validation_simplified_rg_no_others/" \
  --config "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/code/configs/prerender.yaml" \
  --n-jobs 8
