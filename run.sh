# !/bin/bash
NUM_STAGE=5
RUNS=1
LM_LR=2e-5


DATASET='cora'
DROPOUT=0.0
LOGDIR=admm_v2/${DATASET}
mkdir -p $LOGDIR
GNN_LR=1e-3
GNN_DROPOUT=0.0
BETA=1e-2
THETA=1e-4
for ((seed = 0; seed < RUNS; seed++)); do
  rm -rf output
  LOG_PATH=${LOGDIR}/$(date +%Y%m%d_%H%M%S)_seed${seed}.txt
  python -m core.GNNs.trainZ --stage 0 --dataset $DATASET --theta $THETA >> $LOG_PATH ;
  python -m core.GNNs.trainGamma --stage 0 --dataset $DATASET >> $LOG_PATH ;
  for ((i = 1; i < NUM_STAGE; i++)); do
    python -m core.GNNs.trainAdmmGNN --stage $i --dataset $DATASET --dropout $DROPOUT --beta $BETA --lr $GNN_LR --dropout $GNN_DROPOUT >> $LOG_PATH ;
    python -m core.LMs.trainAdmmLM --stage $i --dataset $DATASET --lr $LM_LR >> $LOG_PATH ;
    python -m core.GNNs.trainZ --stage $i --dataset $DATASET --theta $THETA >> $LOG_PATH ;
    python -m core.GNNs.trainGamma --stage $i --dataset $DATASET >> $LOG_PATH ;
  done
done



DATASET='citeseer'
DROPOUT=0.0
LOGDIR=admm_v2/${DATASET}
mkdir -p $LOGDIR
GNN_LR=1e-3
GNN_DROPOUT=0.0
BETA=1e-2
THETA=1e-3
for ((seed = 0; seed < RUNS; seed++)); do
  rm -rf output
  LOG_PATH=${LOGDIR}/$(date +%Y%m%d_%H%M%S)_seed${seed}.txt
  python -m core.GNNs.trainZ --stage 0 --dataset $DATASET --theta $THETA >> $LOG_PATH ;
  python -m core.GNNs.trainGamma --stage 0 --dataset $DATASET >> $LOG_PATH ;
  for ((i = 1; i < NUM_STAGE; i++)); do
    python -m core.GNNs.trainAdmmGNN --stage $i --dataset $DATASET --dropout $DROPOUT --beta $BETA --lr $GNN_LR --dropout $GNN_DROPOUT >> $LOG_PATH ;
    python -m core.LMs.trainAdmmLM --stage $i --dataset $DATASET --lr $LM_LR >> $LOG_PATH ;
    python -m core.GNNs.trainZ --stage $i --dataset $DATASET --theta $THETA >> $LOG_PATH ;
    python -m core.GNNs.trainGamma --stage $i --dataset $DATASET >> $LOG_PATH ;
  done
done


DATASET='pubmed'
DROPOUT=0.0
LOGDIR=admm_v2/${DATASET}
mkdir -p $LOGDIR
GNN_LR=1e-3
GNN_DROPOUT=0.0
BETA=1e-3
THETA=1e-5
for ((seed = 0; seed < RUNS; seed++)); do
  rm -rf output
  LOG_PATH=${LOGDIR}/$(date +%Y%m%d_%H%M%S)_seed${seed}.txt
  python -m core.GNNs.trainZ --stage 0 --dataset $DATASET --theta $THETA >> $LOG_PATH ;
  python -m core.GNNs.trainGamma --stage 0 --dataset $DATASET >> $LOG_PATH ;
  for ((i = 1; i < NUM_STAGE; i++)); do
    python -m core.GNNs.trainAdmmGNN --stage $i --dataset $DATASET --dropout $DROPOUT --beta $BETA --lr $GNN_LR --dropout $GNN_DROPOUT >> $LOG_PATH ;
    python -m core.LMs.trainAdmmLM --stage $i --dataset $DATASET --lr $LM_LR >> $LOG_PATH ;
    python -m core.GNNs.trainZ --stage $i --dataset $DATASET --theta $THETA >> $LOG_PATH ;
    python -m core.GNNs.trainGamma --stage $i --dataset $DATASET >> $LOG_PATH ;
  done
done


DATASET='ogbn-arxiv'
DROPOUT=0.0
LOGDIR=admm_v2/${DATASET}
mkdir -p $LOGDIR
GNN_LR=1e-2
GNN_DROPOUT=0.5
BETA=1e-6
THETA=1e-7
for ((seed = 0; seed < RUNS; seed++)); do
  rm -rf output
  LOG_PATH=${LOGDIR}/$(date +%Y%m%d_%H%M%S)_seed${seed}.txt
  python -m core.GNNs.trainZ --stage 0 --dataset $DATASET --theta $THETA >> $LOG_PATH ;
  python -m core.GNNs.trainGamma --stage 0 --dataset $DATASET >> $LOG_PATH ;
  for ((i = 1; i < NUM_STAGE; i++)); do
    python -m core.GNNs.trainAdmmGNN --stage $i --dataset $DATASET --dropout $DROPOUT --beta $BETA --lr $GNN_LR --dropout $GNN_DROPOUT >> $LOG_PATH ;
    python -m core.LMs.trainAdmmLM --stage $i --dataset $DATASET --lr $LM_LR >> $LOG_PATH ;
    python -m core.GNNs.trainZ --stage $i --dataset $DATASET --theta $THETA >> $LOG_PATH ;
    python -m core.GNNs.trainGamma --stage $i --dataset $DATASET >> $LOG_PATH ;
  done
done