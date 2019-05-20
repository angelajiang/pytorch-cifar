expname=$1
SAMPLING_MIN=$2
NET=$3
BATCH_SIZE=$4
START_EPOCH=$5
ERROR_RATE=$6

NUM_TRIALS=1

set -x

ulimit -n 2048
ulimit -a

EXP_PREFIX=$expname"_shuffle"$ERROR_RATE
SAMPLING_STRATEGY="sampling"
LR="data/config/lr_sched_fast"
DECAY=0.0005
MAX_NUM_BACKPROPS=5840000
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

  OUTPUT_FILE="sampling_cifar10_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_$HIST_SIZE_"$DECAY"_trial"$i"_seed"$SEED"_v2"
  PICKLE_PREFIX="sampling_cifar10_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_$HIST_SIZE_"$DECAY"_trial"$i"_seed"$SEED

  echo $OUTPUT_DIR/$OUTPUT_FILE

  time python main.py \
    --sb-strategy=$SAMPLING_STRATEGY \
    --prob-strategy="relative-cubed" \
    --no-logging \
    --sb-start-epoch=$START_EPOCH \
    --net=$NET \
    --batch-size=$BATCH_SIZE \
    --decay=$DECAY \
    --max-num-backprops=$MAX_NUM_BACKPROPS \
    --pickle-dir=$PICKLE_DIR \
    --pickle-prefix=$PICKLE_PREFIX \
    --sampling-min=$SAMPLING_MIN \
    --augment \
    --seed=$SEED \
    --randomize-labels=$ERROR_RATE \
    --lr-sched $LR &> $OUTPUT_DIR/$OUTPUT_FILE

  let "SEED=SEED+1"
done

