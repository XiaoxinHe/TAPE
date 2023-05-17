# TAG

# Python environment setup with Conda
```
conda create --name tag python=3.8
conda activate tag

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c pyg pytorch-sparse
conda install -c pyg pytorch-scatter
conda install -c pyg pytorch-cluster
conda install -c pyg pyg
pip install ogb
pip install transformers
```

# Fine-tuning the language models
```
# To train the LM on the original text attributes, i.e., title and abstract
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.LMs.trainLM

# To train the LM on the explanations generated by ChatGPT
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.LMs.trainLM --use_gpt
```


# Training the GNN models
```
# To train GNNs using our enriched features
python -m core.GNNs.trainEnsemble --gnn_model_name GCN/SAGE
python -m core.GNNs.trainEnsemble --gnn_model_name RevGAT --use_dgl --lr 0.002 --dropout 0.75

# To train GNNs using OGB features, use "--use_ogb"
python -m core.GNNs.trainGNN --gnn_model_name GCN/SAGE --use_ogb
python -m core.GNNs.trainGNN --gnn_model_name RevGAT --use_dgl --lr 0.002 --dropout 0.75 --use_ogb
```
