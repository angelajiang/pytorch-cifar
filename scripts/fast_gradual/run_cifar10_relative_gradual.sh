expname=$1
SAMPLING_MIN=$2
NET=$3
BATCH_SIZE=$4
PROB_STRATEGY=$5
LOSS=$6
MAX_HISTORY_LEN=$7

NUM_TRIALS=1

set -x

ulimit -n 2048
ulimit -a

EXP_PREFIX=$expname
SAMPLING_STRATEGY="sampling"
LR="data/config/lr_sched_fast_gradual"
DECAY=0.0005
MAX_NUM_BACKPROPS=5500000
SEED=1337

EXP_NAME=$EXP_PREFIX"_"$PROB_STRATEGY"_"$LOSS

mkdir "/proj/BigLearning/ahjiang/output/cifar10/"
OUTPUT_DIR="/proj/BigLearning/ahjiang/output/cifar10/"$EXP_NAME
PICKLE_DIR=$OUTPUT_DIR/pickles
mkdir $OUTPUT_DIR
mkdir $PICKLE_DIR

git rev-parse HEAD &> $OUTPUT_DIR/sha

for i in `seq 1 $NUM_TRIALS`
do

  OUTPUT_FILE="sampling_cifar10_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_"$MAX_HISTORY_LEN"_"$DECAY"_trial"$i"_seed"$SEED"_v2"
  PICKLE_PREFIX="sampling_cifar10_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_"$MAX_HISTORY_LEN"_"$DECAY"_trial"$i"_seed"$SEED

  echo $OUTPUT_DIR/$OUTPUT_FILE

  time python main.py \
    --prob-strategy=$PROB_STRATEGY \
    --max-history-len=$MAX_HISTORY_LEN \
    --prob-loss-fn=$LOSS \
    --sb-start-epoch=1 \
    --sb-strategy=$SAMPLING_STRATEGY \
    --net=$NET \
    --batch-size=$BATCH_SIZE \
    --decay=$DECAY \
    --max-num-backprops=$MAX_NUM_BACKPROPS \
    --pickle-dir=$PICKLE_DIR \
    --pickle-prefix=$PICKLE_PREFIX \
    --sampling-min=$SAMPLING_MIN \
    --augment \
    --seed=$SEED \
    --lr-sched $LR &> $OUTPUT_DIR/$OUTPUT_FILE

  let "SEED=SEED+1"
done
