set -x

ulimit -n 2048
ulimit -a

EXP_NAME=$1
SAMPLING_MIN=$2
START_EPOCH=$3

NET="resnet"
BATCH_SIZE=128
NUM_TRIALS=3
SAMPLING_STRATEGY="sampling"
LR="data/config/lr_sched_svhn"
DECAY=0.0005
MAX_NUM_BACKPROPS=17500000
SEED=1337

mkdir "/proj/BigLearning/ahjiang/output/svhn/"
OUTPUT_DIR="/proj/BigLearning/ahjiang/output/svhn/"$EXP_NAME
PICKLE_DIR=$OUTPUT_DIR/pickles
mkdir $OUTPUT_DIR
mkdir $PICKLE_DIR

git rev-parse HEAD &> $OUTPUT_DIR/sha

for i in `seq 1 $NUM_TRIALS`
do

  OUTPUT_FILE="sampling_svhn_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_0.0_"$DECAY"_trial"$i"_seed"$SEED"_v2"
  PICKLE_PREFIX="sampling_svhn_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_0.0_"$DECAY"_trial"$i"_seed"$SEED

  echo $OUTPUT_DIR/$OUTPUT_FILE

  time python main.py \
    --sb-strategy=$SAMPLING_STRATEGY \
    --sb-start-epoch=$START_EPOCH \
    --net=$NET \
    --batch-size=$BATCH_SIZE \
    --decay=$DECAY \
    --max-num-backprops=$MAX_NUM_BACKPROPS \
    --pickle-dir=$PICKLE_DIR \
    --pickle-prefix=$PICKLE_PREFIX \
    --sampling-min=$SAMPLING_MIN \
    --dataset="svhn" \
    --augment \
    --seed=$SEED \
    --lr-sched $LR &> $OUTPUT_DIR/$OUTPUT_FILE

  let "SEED=SEED+1"
done
