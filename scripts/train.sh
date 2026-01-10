python train.py \
    --data_root data/Flickr8k_Dataset \
    --output_dir checkpoints \
    --epochs 20 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --warmup_epochs 2 \
    --gradient_accumulation_steps 2