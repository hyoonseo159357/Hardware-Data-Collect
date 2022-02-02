#!bin/bash
INSTANCE_TYPE=$1
MODEL_NAME=$2
BATCH_SIZE=$3
DATASET_SIZE=$4

#Training CMD
TRAIN_CMD="/home/ubuntu/Hardware-Data2/workload.py \
--model $MODEL_NAME --dataset $DATASET_SIZE --batch_size $BATCH_SIZE --instance_type $INSTANCE_TYPE"

sudo -i -u root bash << EOF
python3.6 $TRAIN_CMD
EOF
