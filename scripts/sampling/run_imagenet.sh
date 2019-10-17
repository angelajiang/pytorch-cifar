expname=$1
SAMPLING_MIN=$2
NET="resnet"
BATCH_SIZE=128
START_EPOCH=1

NUM_TRIALS=1

set -x

ulimit -n 2048
ulimit -a

EXP_PREFIX=$expname
SAMPLING_STRATEGY="sampling"
LR="data/config/lr_sched_imagenet_gradual"
DECAY=0.0001
MAX_NUM_BACKPROPS=25802100
SEED=1337
DATA_DIR="/proj/BigLearning/ahjiang/datasets/imagenet-data"

EXP_NAME=$EXP_PREFIX

mkdir "/proj/BigLearning/ahjiang/output/imagenet/"
OUTPUT_DIR="/proj/BigLearning/ahjiang/output/imagenet/"$EXP_NAME
PICKLE_DIR=$OUTPUT_DIR/pickles
mkdir $OUTPUT_DIR
mkdir $PICKLE_DIR

git rev-parse HEAD &> $OUTPUT_DIR/sha

for i in `seq 1 $NUM_TRIALS`
do

  OUTPUT_FILE="sampling_imagenet_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_0.0_"$DECAY"_trial"$i"_seed"$SEED"_v2"
  PICKLE_PREFIX="sampling_imagenet_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_0.0_"$DECAY"_trial"$i"_seed"$SEED

  echo $OUTPUT_DIR/$OUTPUT_FILE

  python imagenet.py \
    --strategy=logbias \
    --dataset=imagenet \
    --datadir=$DATA_DIR \
    --net=$NET \
    --batch-size=$BATCH_SIZE \
    --decay=$DECAY \
    --augment \
    --seed=$SEED \
    --lr-sched $LR &> $OUTPUT_DIR/$OUTPUT_FILE

  let "SEED=SEED+1"
done
