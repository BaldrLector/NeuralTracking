
# Name of the train and val datasets
train_dir="/media/baldr/新加卷/deepdeform_v1_1/train"
val_dir="/media/baldr/新加卷/deepdeform_v1_1/val"

# Give a name to your experiment
experiment="debug_flownet"
echo ${experiment}

GPU=${1:-0}

CUDA_VISIBLE_DEVICES=${GPU} python train.py --train_dir="${train_dir}" \
                                            --val_dir="${val_dir}" \
                                            --experiment="${experiment}"