# !/bin/bash
NUM_STAGE=5
NUM_LAYERS=2


DATASET='cora'
LOG_PATH=$(date +%Y%m%d_%H%M%S)
DROPOUT=0.0
python -m core.LMs.trainAdmmLM --stage 0 --dataset $DATASET >> admm/$DATASET/${LOG_PATH}.txt ;
python -m core.GNNs.trainZ --stage 0 --dataset $DATASET --gnn_num_layers $NUM_LAYERS --gnn_dropout $DROPOUT >> admm/$DATASET/${LOG_PATH}.txt ;
python -m core.GNNs.trainGamma --stage 0 --dataset $DATASET >> admm/$DATASET/${LOG_PATH}.txt ;
for ((i = 1; i <= NUM_STAGE; i++)); do
  sleep 1
  python -m core.GNNs.trainAdmmGNN --stage $i --dataset $DATASET --num_layers $NUM_LAYERS --dropout $DROPOUT >> admm/$DATASET/${LOG_PATH}.txt ;
  python -m core.LMs.trainAdmmLM --stage $i --dataset $DATASET >> admm/$DATASET/${LOG_PATH}.txt ;
  python -m core.GNNs.trainZ --stage $i --dataset $DATASET --gnn_num_layers $NUM_LAYERS --gnn_dropout $DROPOUT >> admm/$DATASET/${LOG_PATH}.txt ;
  python -m core.GNNs.trainGamma --stage $i --dataset $DATASET >> admm/$DATASET/${LOG_PATH}.txt ;
done

DATASET='citeseer'
LOG_PATH=$(date +%Y%m%d_%H%M%S)
DROPOUT=0.0
python -m core.LMs.trainAdmmLM --stage 0 --dataset $DATASET >> admm/$DATASET/${LOG_PATH}.txt ;
python -m core.GNNs.trainZ --stage 0 --dataset $DATASET --gnn_num_layers $NUM_LAYERS --gnn_dropout $DROPOUT >> admm/$DATASET/${LOG_PATH}.txt ;
python -m core.GNNs.trainGamma --stage 0 --dataset $DATASET >> admm/$DATASET/${LOG_PATH}.txt ;
for ((i = 1; i <= NUM_STAGE; i++)); do
  sleep 1
  python -m core.GNNs.trainAdmmGNN --stage $i --dataset $DATASET --num_layers $NUM_LAYERS --dropout $DROPOUT >> admm/$DATASET/${LOG_PATH}.txt ;
  python -m core.LMs.trainAdmmLM --stage $i --dataset $DATASET >> admm/$DATASET/${LOG_PATH}.txt ;
  python -m core.GNNs.trainZ --stage $i --dataset $DATASET --gnn_num_layers $NUM_LAYERS --gnn_dropout $DROPOUT >> admm/$DATASET/${LOG_PATH}.txt ;
  python -m core.GNNs.trainGamma --stage $i --dataset $DATASET >> admm/$DATASET/${LOG_PATH}.txt ;
done

DATASET='pubmed'
LOG_PATH=$(date +%Y%m%d_%H%M%S)
DROPOUT=0.0
python -m core.LMs.trainAdmmLM --stage 0 --dataset $DATASET >> admm/$DATASET/${LOG_PATH}.txt ;
python -m core.GNNs.trainZ --stage 0 --dataset $DATASET --gnn_num_layers $NUM_LAYERS --gnn_dropout $DROPOUT >> admm/$DATASET/${LOG_PATH}.txt ;
python -m core.GNNs.trainGamma --stage 0 --dataset $DATASET >> admm/$DATASET/${LOG_PATH}.txt ;
for ((i = 1; i <= NUM_STAGE; i++)); do
  sleep 1
  python -m core.GNNs.trainAdmmGNN --stage $i --dataset $DATASET --num_layers $NUM_LAYERS --dropout $DROPOUT >> admm/$DATASET/${LOG_PATH}.txt ;
  python -m core.LMs.trainAdmmLM --stage $i --dataset $DATASET >> admm/$DATASET/${LOG_PATH}.txt ;
  python -m core.GNNs.trainZ --stage $i --dataset $DATASET --gnn_num_layers $NUM_LAYERS --gnn_dropout $DROPOUT >> admm/$DATASET/${LOG_PATH}.txt ;
  python -m core.GNNs.trainGamma --stage $i --dataset $DATASET >> admm/$DATASET/${LOG_PATH}.txt ;
done



DATASET='ogbn-arxiv'
LOG_PATH=$(date +%Y%m%d_%H%M%S)
DROPOUT=0.5
python -m core.LMs.trainAdmmLM --stage 0 --dataset $DATASET >> admm/$DATASET/${LOG_PATH}.txt ;
python -m core.GNNs.trainZ --stage 0 --dataset $DATASET --gnn_num_layers $NUM_LAYERS --gnn_dropout $DROPOUT >> admm/$DATASET/${LOG_PATH}.txt ;
python -m core.GNNs.trainGamma --stage 0 --dataset $DATASET >> admm/$DATASET/${LOG_PATH}.txt ;
for ((i = 1; i <= NUM_STAGE; i++)); do
  sleep 1
  python -m core.GNNs.trainAdmmGNN --stage $i --dataset $DATASET --num_layers $NUM_LAYERS --dropout $DROPOUT >> admm/$DATASET/${LOG_PATH}.txt ;
  python -m core.LMs.trainAdmmLM --stage $i --dataset $DATASET >> admm/$DATASET/${LOG_PATH}.txt ;
  python -m core.GNNs.trainZ --stage $i --dataset $DATASET --gnn_num_layers $NUM_LAYERS --gnn_dropout $DROPOUT >> admm/$DATASET/${LOG_PATH}.txt ;
  python -m core.GNNs.trainGamma --stage $i --dataset $DATASET >> admm/$DATASET/${LOG_PATH}.txt ;
done


exit

