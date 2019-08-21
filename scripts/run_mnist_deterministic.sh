set -x

ulimit -n 2048
ulimit -a

EXP_NAME=$1
BATCH_SIZE=$2
LR=$3
MAX_NUM_BACKPROPS=$4
SAMPLING_MIN=$5
SEED=1337

DECAY=0.0005
NET="lecunn"

mkdir "/proj/BigLearning/ahjiang/output/mnist/"
OUTPUT_DIR="/proj/BigLearning/ahjiang/output/mnist/"$EXP_NAME
PICKLE_DIR=$OUTPUT_DIR/pickles
mkdir $OUTPUT_DIR
mkdir $PICKLE_DIR

git rev-parse HEAD &> $OUTPUT_DIR/sha

NUM_TRIALS=1
for i in `seq 1 $NUM_TRIALS`
do
  OUTPUT_FILE="deterministic_cifar10_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_"$LR"_"$DECAY"_trial"$i"_seed"$SEED"_v2"
  PICKLE_PREFIX="deterministic_cifar10_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_"$LR"_"$DECAY"_trial"$i"_seed"$SEED

  echo $OUTPUT_DIR/$OUTPUT_FILE

  time python main.py \
    --sb-strategy=deterministic \
    --dataset=mnist \
    --batch-size=$BATCH_SIZE \
    --decay=$DECAY \
    --max-num-backprops=$MAX_NUM_BACKPROPS \
    --pickle-dir=$PICKLE_DIR \
    --pickle-prefix=$PICKLE_PREFIX \
    --sampling-min=$SAMPLING_MIN \
    --seed=$SEED \
    --lr $LR &> $OUTPUT_DIR/$OUTPUT_FILE

  let "SEED=SEED+1"
  echo $SEED" should be 1338"
done
