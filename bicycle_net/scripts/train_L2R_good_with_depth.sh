set -ex
MODEL='bicycle_gan'
# dataset details
CLASS='L2R_good_with_depth' # facades, day2night, edges2shoes, edges2handbags, maps
BATCH_SIZE=4
NZ=32
NO_FLIP=''
DIRECTION='AtoB'
LOAD_SIZE_W=512
LOAD_SIZE_H=256
PREPROCESS='scale_width_and_crop'
CROP_SIZE_W=512
CROP_SIZE_H=256
INPUT_NC=3
NITER=0
NITER_DECAY=5
SAVE_EPOCH=1
NGF=96
NEF=96
NDF=96

# training
GPU_ID=0
DISPLAY_ID=$((GPU_ID*10+1))
CHECKPOINTS_DIR=./checkpoints/${CLASS}/
NAME=${CLASS}_${MODEL}

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./train.py \
  --display_id ${DISPLAY_ID} \
  --dataroot ./datasets/${CLASS} \
  --name ${NAME} \
  --model ${MODEL} \
  --direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --batch_size ${BATCH_SIZE} \
  --load_size_w ${LOAD_SIZE_W} \
  --load_size_h ${LOAD_SIZE_H} \
  --preprocess ${PREPROCESS} \
  --crop_size_w ${CROP_SIZE_W} \
  --crop_size_h ${CROP_SIZE_H} \
  --nz ${NZ} \
  --ngf ${NGF} \
  --nef ${NEF} \
  --ndf ${NDF} \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --save_epoch_freq ${SAVE_EPOCH} \
  --use_dropout \
  --continue_train \
  --lr 0.00008 \
  --lambda_P 1.0 \
  --epoch_count 1
