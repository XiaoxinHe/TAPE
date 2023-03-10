# !/bin/bash
NUM_STAGE=5
RUNS=4
LM_LR=3e-5

DATASET='cora'
DROPOUT=0.0
LOGDIR=admm/${DATASET}
mkdir -p $LOGDIR
for ((seed = 0; seed < RUNS; seed++)); do
  LOG_PATH=${LOGDIR}/$(date +%Y%m%d_%H%M%S)_seed${seed}.txt
  python -m core.GNNs.trainZ --stage 0 --dataset $DATASET >> $LOG_PATH ;
  python -m core.GNNs.trainGamma --stage 0 --dataset $DATASET >> $LOG_PATH ;
  for ((i = 1; i < NUM_STAGE; i++)); do
    python -m core.GNNs.trainAdmmGNN --stage $i --dataset $DATASET --dropout $DROPOUT >> $LOG_PATH ;
    python -m core.LMs.trainAdmmLM --stage $i --dataset $DATASET --lr $LM_LR >> $LOG_PATH ;
    python -m core.LMs.trainAdmmLMClassifier --stage $i --dataset $DATASET --lr $LM_LR >> $LOG_PATH ;
    python -m core.GNNs.trainZ --stage $i --dataset $DATASET >> $LOG_PATH ;
    python -m core.GNNs.trainGamma --stage $i --dataset $DATASET >> $LOG_PATH ;
  done
done


DATASET='citeseer'
DROPOUT=0.0
LOGDIR=admm/${DATASET}
mkdir -p $LOGDIR
for ((seed = 0; seed < RUNS; seed++)); do
  LOG_PATH=${LOGDIR}/$(date +%Y%m%d_%H%M%S)_seed${seed}.txt
  python -m core.GNNs.trainZ --stage 0 --dataset $DATASET >> $LOG_PATH ;
  python -m core.GNNs.trainGamma --stage 0 --dataset $DATASET >> $LOG_PATH ;
  for ((i = 1; i < NUM_STAGE; i++)); do
    python -m core.GNNs.trainAdmmGNN --stage $i --dataset $DATASET --dropout $DROPOUT >> $LOG_PATH ;
    python -m core.LMs.trainAdmmLM --stage $i --dataset $DATASET --lr $LM_LR >> $LOG_PATH ;
    python -m core.LMs.trainAdmmLMClassifier --stage $i --dataset $DATASET --lr $LM_LR >> $LOG_PATH ;
    python -m core.GNNs.trainZ --stage $i --dataset $DATASET >> $LOG_PATH ;
    python -m core.GNNs.trainGamma --stage $i --dataset $DATASET >> $LOG_PATH ;
  done
done


DATASET='pubmed'
DROPOUT=0.0
LOGDIR=admm/${DATASET}
mkdir -p $LOGDIR
for ((seed = 0; seed < RUNS; seed++)); do
  LOG_PATH=${LOGDIR}/$(date +%Y%m%d_%H%M%S)_seed${seed}.txt
  python -m core.GNNs.trainZ --stage 0 --dataset $DATASET >> $LOG_PATH ;
  python -m core.GNNs.trainGamma --stage 0 --dataset $DATASET >> $LOG_PATH ;
  for ((i = 1; i < NUM_STAGE; i++)); do
    python -m core.GNNs.trainAdmmGNN --stage $i --dataset $DATASET --dropout $DROPOUT >> $LOG_PATH ;
    python -m core.LMs.trainAdmmLM --stage $i --dataset $DATASET --lr $LM_LR >> $LOG_PATH ;
    python -m core.LMs.trainAdmmLMClassifier --stage $i --dataset $DATASET --lr $LM_LR >> $LOG_PATH ;
    python -m core.GNNs.trainZ --stage $i --dataset $DATASET >> $LOG_PATH ;
    python -m core.GNNs.trainGamma --stage $i --dataset $DATASET >> $LOG_PATH ;
  done
done


DATASET='ogbn-arxiv'
DROPOUT=0.5
LOGDIR=admm/${DATASET}
mkdir -p $LOGDIR
for ((seed = 0; seed < RUNS; seed++)); do
  LOG_PATH=${LOGDIR}/$(date +%Y%m%d_%H%M%S)_seed${seed}.txt
  python -m core.GNNs.trainZ --stage 0 --dataset $DATASET >> $LOG_PATH ;
  python -m core.GNNs.trainGamma --stage 0 --dataset $DATASET >> $LOG_PATH ;
  for ((i = 1; i < NUM_STAGE; i++)); do
    python -m core.GNNs.trainAdmmGNN --stage $i --dataset $DATASET --dropout $DROPOUT >> $LOG_PATH ;
    python -m core.LMs.trainAdmmLM --stage $i --dataset $DATASET --lr $LM_LR >> $LOG_PATH ;
    python -m core.LMs.trainAdmmLMClassifier --stage $i --dataset $DATASET --lr $LM_LR >> $LOG_PATH ;
    python -m core.GNNs.trainZ --stage $i --dataset $DATASET >> $LOG_PATH ;
    python -m core.GNNs.trainGamma --stage $i --dataset $DATASET >> $LOG_PATH ;
  done
done

