CUDA_VISIBLE_DEVICES=-1 \
python /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/prerender/prerender.py \
  --data-path "/home/manolotis/sandbox/waymoMotion/data/reduced/tf_example/training/" \
  --output-path "/home/manolotis/sandbox/robustness_benchmark/multipathPP/data/prerendered/training_noisy_heading/" \
  --config "/home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/prerender_noisy_heading.yaml" \
  --n-jobs 8