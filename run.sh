for dataset in 'ogbn-arxiv' 'pubmed' 'cora'
do
    for seed in 0 1 2 3
    do
    python -m core.LMs.trainLM --dataset $dataset --seed $seed >> ${dataset}_lm.out
    python -m core.LMs.trainLM --dataset $dataset --seed $seed --use_gpt >> ${dataset}_lm.out
    done
    python -m core.GNNs.trainEnsemble --dataset_name $dataset --gnn_model_name GCN >> ${dataset}_gnn.out
    python -m core.GNNs.trainEnsemble --dataset_name $dataset --gnn_model_name SAGE >> ${dataset}_gnn.out
    python -m core.GNNs.trainEnsemble --dataset_name $dataset --gnn_model_name RevGAT --use_dgl --lr 0.002 --dropout 0.002 >> ${dataset}_gnn.out    
done

