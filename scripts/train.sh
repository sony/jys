#########
# Train #
#########
lr=0.001
batch_size=2048
python3 train.py \
    --dataset_name "count"          \
    --model_name "sedd"             \
    --arch_name "ddit"              \
    --scheduler_name "tweedie"      \
    --noise_schedule "loglinear"    \
    --train_iter 1024               \
    --batch_size $batch_size        \
    --lr $lr                        \
    --wandb_key ""                  \
    --device cuda:0

# move the trained model to the weights directory
mkdir -p weights
mkdir -p weights/count-sedd
mv runs/count-sedd/state_dict-$lr-$batch_size.pt weights/count-sedd/ckpt.pt
rm -rf runs/count-sedd