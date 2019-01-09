expname=$1
SAMPLING_MIN=$2
NET=$3
BATCH_SIZE=$4
SEED=$5
TRIAL=$6

set -x

ulimit -n 2048
ulimit -a

EXP_PREFIX=$expname
SAMPLING_STRATEGY="deterministic"
LR="data/config/lr_sched_orig"
DECAY=0.0005
MAX_NUM_BACKPROPS=17500000

EXP_NAME=$EXP_PREFIX

mkdir "/proj/BigLearning/ahjiang/output/cifar10/"
OUTPUT_DIR="/proj/BigLearning/ahjiang/output/cifar10/"$EXP_NAME
PICKLE_DIR=$OUTPUT_DIR/pickles
mkdir $OUTPUT_DIR
mkdir $PICKLE_DIR

git rev-parse HEAD &> $OUTPUT_DIR/sha

OUTPUT_FILE="deterministic_cifar10_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_0.0_"$DECAY"_trial"$TRIAL"_seed"$SEED"_v2"
PICKLE_PREFIX="deterministic_cifar10_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_0.0_"$DECAY"_trial"$TRIAL"_seed"$SEED

echo $OUTPUT_DIR/$OUTPUT_FILE

time python main.py \
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
