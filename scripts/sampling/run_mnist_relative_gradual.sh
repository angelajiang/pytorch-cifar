expname=$1
SAMPLING_MIN=$2
BATCH_SIZE=$3
PROB_STRATEGY=$4
LOSS=$5
MAX_HISTORY_LEN=$6

NUM_TRIALS=1

set -x

ulimit -n 2048
ulimit -a

EXP_PREFIX=$expname
SAMPLING_STRATEGY="sampling"
LR=0.001
DECAY=0.0005
MAX_NUM_BACKPROPS=1000000
SEED=1337
NET=LeNet-5

EXP_NAME=$EXP_PREFIX"_"$PROB_STRATEGY"_"$LOSS

mkdir "/proj/BigLearning/ahjiang/output/mnist/"
OUTPUT_DIR="/proj/BigLearning/ahjiang/output/mnist/"$EXP_NAME
PICKLE_DIR=$OUTPUT_DIR/pickles
mkdir $OUTPUT_DIR
mkdir $PICKLE_DIR

git rev-parse HEAD &> $OUTPUT_DIR/sha

for i in `seq 1 $NUM_TRIALS`
do

  OUTPUT_FILE="sampling_mnist_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_"$MAX_HISTORY_LEN"_"$DECAY"_trial"$i"_seed"$SEED"_v2"
  PICKLE_PREFIX="sampling_mnist_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_"$MAX_HISTORY_LEN"_"$DECAY"_trial"$i"_seed"$SEED

  echo $OUTPUT_DIR/$OUTPUT_FILE

  time python main.py \
    --prob-strategy=$PROB_STRATEGY \
    --max-history-len=$MAX_HISTORY_LEN \
    --dataset=mnist \
    --prob-loss-fn=$LOSS \
    --sb-start-epoch=1 \
    --sb-strategy=$SAMPLING_STRATEGY \
    --batch-size=$BATCH_SIZE \
    --decay=$DECAY \
    --max-num-backprops=$MAX_NUM_BACKPROPS \
    --pickle-dir=$PICKLE_DIR \
    --pickle-prefix=$PICKLE_PREFIX \
    --sampling-min=$SAMPLING_MIN \
    --augment \
    --seed=$SEED \
    --lr $LR &> $OUTPUT_DIR/$OUTPUT_FILE

  let "SEED=SEED+1"
done
