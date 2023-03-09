# !/bin/bash
NUM_STAGE=5
RUNS=4

DATASET='cora'
DROPOUT=0.0
for ((seed = 0; seed < RUNS; seed++)); do
  LOG_PATH=kd/${DATASET}/$(date +%Y%m%d_%H%M%S)_seed${seed}.txt
  mkdir -p log/$LOG_PATH
  python -m core.GNNs.trainKDGNN --stage 0 --dataset $DATASET --dropout $DROPOUT >> $LOG_PATH ;
  for ((i = 1; i < NUM_STAGE; i++)); do
    sleep 1
    python -m core.LMs.trainKDLM --stage $i --dataset $DATASET --seed $seed >> $LOG_PATH ;
    python -m core.GNNs.trainKDGNN --stage $i --dataset $DATASET --dropout $DROPOUT >> $LOG_PATH ;
  done
done

DATASET='citeseer'
DROPOUT=0.0
for ((seed = 0; seed < RUNS; seed++)); do
  LOG_PATH=kd/${DATASET}/$(date +%Y%m%d_%H%M%S)_seed${seed}.txt
  mkdir -p log/$LOG_PATH
  python -m core.GNNs.trainKDGNN --stage 0 --dataset $DATASET --dropout $DROPOUT >> $LOG_PATH ;
  for ((i = 1; i < NUM_STAGE; i++)); do
    sleep 1
    python -m core.LMs.trainKDLM --stage $i --dataset $DATASET --seed $seed >> $LOG_PATH ;
    python -m core.GNNs.trainKDGNN --stage $i --dataset $DATASET --dropout $DROPOUT >> $LOG_PATH ;
  done
done

DATASET='pubmed'
DROPOUT=0.0
for ((seed = 0; seed < RUNS; seed++)); do
  LOG_PATH=kd/${DATASET}/$(date +%Y%m%d_%H%M%S)_seed${seed}.txt
  mkdir -p log/$LOG_PATH
  python -m core.GNNs.trainKDGNN --stage 0 --dataset $DATASET --dropout $DROPOUT >> $LOG_PATH ;
  for ((i = 1; i < NUM_STAGE; i++)); do
    sleep 1
    python -m core.LMs.trainKDLM --stage $i --dataset $DATASET --seed $seed >> $LOG_PATH ;
    python -m core.GNNs.trainKDGNN --stage $i --dataset $DATASET --dropout $DROPOUT >> $LOG_PATH ;
  done
done

DATASET='ogbn-arxiv'
DROPOUT=0.5
for ((seed = 0; seed < RUNS; seed++)); do
  LOG_PATH=kd/${DATASET}/$(date +%Y%m%d_%H%M%S)_seed${seed}.txt
  mkdir -p log/$LOG_PATH
  python -m core.GNNs.trainKDGNN --stage 0 --dataset $DATASET --dropout $DROPOUT >> $LOG_PATH ;
  for ((i = 1; i < NUM_STAGE; i++)); do
    sleep 1
    python -m core.LMs.trainKDLM --stage $i --dataset $DATASET --seed $seed >> $LOG_PATH ;
    python -m core.GNNs.trainKDGNN --stage $i --dataset $DATASET --dropout $DROPOUT >> $LOG_PATH ;
  done
done

exit