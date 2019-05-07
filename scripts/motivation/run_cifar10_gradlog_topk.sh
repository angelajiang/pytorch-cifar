expname=$1
SAMPLE_SIZE=$2
BATCH_SIZE=$3
NET=$4
START_EPOCH=$5
LR=$6

set -x

ulimit -n 2048
ulimit -a

EXP_PREFIX=$expname
SAMPLING_MIN=0
SAMPLING_STRATEGY="topk"
NUM_TRIALS=1
DECAY=0.0005
MAX_NUM_BACKPROPS=3000000
SEED=1337

EXP_NAME=$EXP_PREFIX

mkdir "/proj/BigLearning/ahjiang/output/cifar10/"
OUTPUT_DIR="/proj/BigLearning/ahjiang/output/cifar10/"$EXP_NAME
PICKLE_DIR=$OUTPUT_DIR/pickles
mkdir $OUTPUT_DIR
mkdir $PICKLE_DIR

git rev-parse HEAD &> $OUTPUT_DIR/sha

for i in `seq 1 $NUM_TRIALS`
do

  OUTPUT_FILE=$SAMPLING_STRATEGY"_cifar10_"$NET"_"$SAMPLE_SIZE"_"$BATCH_SIZE"_0.0_"$DECAY"_trial"$i"_seed"$SEED"_v2"
  PICKLE_PREFIX=$SAMPLING_STRATEGY"_cifar10_"$NET"_"$SAMPLE_SIZE"_"$BATCH_SIZE"_0.0_"$DECAY"_trial"$i"_seed"$SEED

  echo $OUTPUT_DIR/$OUTPUT_FILE

  time python main.py \
    --sb-strategy=$SAMPLING_STRATEGY \
    --sb-start-epoch=$START_EPOCH \
    --sample-size=$SAMPLE_SIZE \
    --net=$NET \
    --batch-size=$BATCH_SIZE \
    --decay=$DECAY \
    --max-num-backprops=$MAX_NUM_BACKPROPS \
    --pickle-dir=$PICKLE_DIR \
    --pickle-prefix=$PICKLE_PREFIX \
    --sampling-min=$SAMPLING_MIN \
    --augment \
    --log-bias \
    --seed=$SEED \
    --lr $LR &> $OUTPUT_DIR/$OUTPUT_FILE
done
