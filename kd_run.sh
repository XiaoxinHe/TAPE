# !/bin/bash
NUM_STAGE=5
NUM_LAYERS=2
PL_WEIGHT=0.5

DATASET='cora'
LOG_PATH=$(date +%Y%m%d_%H%M%S)
NUM_LAYERS=2
DROPOUT=0.0
for ((i = 0; i <= NUM_STAGE; i++)); do
  sleep 1
  python -m core.LMs.trainKDLM --stage $i --pl_weight $PL_WEIGHT --dataset $DATASET >> kd/$DATASET/${LOG_PATH}.txt ;
  python -m core.GNNs.trainKDGNN --stage $i --dataset $DATASET --num_layers $NUM_LAYERS --dropout $DROPOUT >> kd/$DATASET/${LOG_PATH}.txt ;
done

DATASET='citeseer'
LOG_PATH=$(date +%Y%m%d_%H%M%S)
NUM_LAYERS=2
DROPOUT=0.0
for ((i = 0; i <= NUM_STAGE; i++)); do
  sleep 1
  python -m core.LMs.trainKDLM --stage $i --pl_weight $PL_WEIGHT --dataset $DATASET >> kd/$DATASET/${LOG_PATH}.txt ;
  python -m core.GNNs.trainKDGNN --stage $i --dataset $DATASET --num_layers $NUM_LAYERS --dropout $DROPOUT >> kd/$DATASET/${LOG_PATH}.txt ;
done

DATASET='pubmed'
LOG_PATH=$(date +%Y%m%d_%H%M%S)
NUM_LAYERS=2
DROPOUT=0.0
for ((i = 0; i <= NUM_STAGE; i++)); do
  sleep 1
  python -m core.LMs.trainKDLM --stage $i --pl_weight $PL_WEIGHT --dataset $DATASET >> kd/$DATASET/${LOG_PATH}.txt ;
  python -m core.GNNs.trainKDGNN --stage $i --dataset $DATASET --num_layers $NUM_LAYERS --dropout $DROPOUT >> kd/$DATASET/${LOG_PATH}.txt ;
done

# DATASET='ogbn-arxiv'
# LOG_PATH=$(date +%Y%m%d_%H%M%S)
# NUM_LAYERS=2
# DROPOUT=0.5
# for ((i = 0; i <= NUM_STAGE; i++)); do
#   sleep 1
#   python -m core.LMs.trainKDLM --stage $i --pl_weight $PL_WEIGHT --dataset $DATASET >> kd/$DATASET/${LOG_PATH}.txt ;
#   python -m core.GNNs.trainKDGNN --stage $i --dataset $DATASET --num_layers $NUM_LAYERS --dropout $DROPOUT >> kd/$DATASET/${LOG_PATH}.txt ;
# done

exit