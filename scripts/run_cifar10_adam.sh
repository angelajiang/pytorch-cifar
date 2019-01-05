expname=$1
SAMPLING_MIN=$2
NET=$3
BATCH_SIZE=$4

NUM_TRIALS=1

set -x

ulimit -n 2048
ulimit -a

EXP_PREFIX=$expname
SAMPLING_STRATEGY="deterministic"
LR=0.01
DECAY=0
MAX_NUM_BACKPROPS=17500000
SEED=1337

EXP_NAME=$EXP_PREFIX"_adam"

mkdir "/proj/BigLearning/ahjiang/output/cifar10/"
OUTPUT_DIR="/proj/BigLearning/ahjiang/output/cifar10/"$EXP_NAME
PICKLE_DIR=$OUTPUT_DIR/pickles
mkdir $OUTPUT_DIR
mkdir $PICKLE_DIR

git rev-parse HEAD &> $OUTPUT_DIR/sha

for i in `seq 1 $NUM_TRIALS`
do

  OUTPUT_FILE="deterministic_cifar10_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_"$LR"_"$DECAY"_trial"$i"_seed"$SEED"_v2"
  PICKLE_PREFIX="deterministic_cifar10_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_"$LR"_"$DECAY"_trial"$i"_seed"$SEED

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
    --optimizer=adam \
    --lr $LR &> $OUTPUT_DIR/$OUTPUT_FILE
done
