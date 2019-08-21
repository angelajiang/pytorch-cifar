sex -x
EXPNAME="190104_trials"
NUM_TRIALS=2
SEED=1000

for i in `seq 1 $NUM_TRIALS`
do
  let "SEED=SEED+1"
  bash scripts/run_cifar10_trials.sh $EXPNAME 0.1 densenet 64 $SEED $i
  bash scripts/run_cifar10_trials.sh $EXPNAME 1 densenet 64 $SEED $i
done
