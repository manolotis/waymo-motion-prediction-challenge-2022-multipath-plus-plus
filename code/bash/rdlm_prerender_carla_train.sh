CUDA_VISIBLE_DEVICES=-1 \
python /home/manolotis/sandbox/multipathpp/code/prerender/prerender.py \
  --data-path "/home/manolotis/sandbox/c4r/data/tfExamples/training" \
  --output-path "/home/manolotis/sandbox/multipathpp/data/prerendered/train/" \
  --config "/home/manolotis/sandbox/multipathpp/code/configs/prerender_carla.yaml" \
  --n-jobs 48