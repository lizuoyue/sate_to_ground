set -ex

CLASS="L2R_aug_good_without_depth"

# models
RESULTS_DIR="./results/${CLASS}"
G_PATH="./checkpoints/"${CLASS}"/"${CLASS}"_bicycle_gan/latest_net_G.pth"
E_PATH="./checkpoints/"${CLASS}"/"${CLASS}"_bicycle_gan/latest_net_E.pth"
CHECKPOINTS_DIR="./checkpoints/"${CLASS}"/"${CLASS}"_bicycle_gan"

# dataset
DIRECTION="AtoB" # from domain A to domain B
LOAD_SIZE_W=512
LOAD_SIZE_H=256
CROP_SIZE_W=512
CROP_SIZE_H=256
INPUT_NC=3 # number of channels in the input image
NZ=96
PREPROCESS="resize_and_crop"

NGF=108
NEF=108
NDF=108

LATEST_DIR=${CHECKPOINTS_DIR}"/"${CLASS}
mkdir -p ${LATEST_DIR}
cp ${G_PATH} ${LATEST_DIR}
cp ${E_PATH} ${LATEST_DIR}

# misc
GPU_ID=-1   # gpu id
NUM_TEST=100 # number of input images duirng test
NUM_SAMPLES=5 # number of samples per input images

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./test_sate_encoder.py \
  --dataroot ./datasets/${CLASS} \
  --results_dir ${RESULTS_DIR} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --name ${CLASS} \
  --direction ${DIRECTION} \
  --load_size_w ${LOAD_SIZE_W} \
  --load_size_h ${LOAD_SIZE_H} \
  --crop_size_w ${CROP_SIZE_W} \
  --crop_size_h ${CROP_SIZE_H} \
  --input_nc ${INPUT_NC} \
  --preprocess ${PREPROCESS} \
  --nz ${NZ} \
  --ngf ${NGF} \
  --nef ${NEF} \
  --ndf ${NDF} \
  --num_test ${NUM_TEST} \
  --n_samples ${NUM_SAMPLES} \
  --center_crop \
  --no_flip \
  --sync \
  --gpu_ids "-1" \
  --fc_input_scale 2 \
  --serial_batches
