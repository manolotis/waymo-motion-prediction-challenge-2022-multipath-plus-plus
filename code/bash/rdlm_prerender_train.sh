CUDA_VISIBLE_DEVICES=-1 \
python /home/manolotis/sandbox/robustness_benchmark/multipathPP/code/prerender/prerender.py \
--data-path "/media/disk1/datasets/waymo/motion v1.0/uncompressed/tf_example/training/" \
--output-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/training/" \
--config "/home/manolotis/sandbox/robustness_benchmark/multipathPP/code/configs/prerender.yaml" \
--n-jobs 48