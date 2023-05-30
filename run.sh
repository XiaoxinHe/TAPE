for dataset in 'cora' 'pubmed' 'ogbn-arxiv'
do
    for seed in 0 1 2 3
    do
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM dataset $dataset seed $seed >> ${dataset}_lm.out
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM dataset $dataset seed $seed lm.train.use_gpt True  >> ${dataset}_lm.out
    done
    python -m core.trainEnsemble dataset $dataset gnn.model.name GCN >> ${dataset}_gnn.out
    python -m core.trainEnsemble dataset $dataset gnn.model.name SAGE >> ${dataset}_gnn.out
    python -m core.trainEnsemble dataset $dataset gnn.model.name RevGAT gnn.train.use_dgl True gnn.train.lr 0.002 gnn.train.dropout 0.5 >> ${dataset}_gnn.out
done