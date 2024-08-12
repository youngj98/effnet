

# conda activate
conda activate timm_py39

# Train model

CUDA_VISIBLE_DEVICES=0 python train_weather.py
CUDA_VISIBLE_DEVICES=1 python train_weather.py
CUDA_VISIBLE_DEVICES=2 python train_weather.py
CUDA_VISIBLE_DEVICES=3 python train_weather.py