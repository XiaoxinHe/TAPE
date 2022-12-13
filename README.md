# Explanations as Features: </br>LLM-Based Features for Text-Attributed Graphs

<img src="./overview.svg">

## 0. Python environment setup with Conda
```
conda create --name TAPE python=3.8
conda activate TAPE

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c pyg pytorch-sparse
conda install -c pyg pytorch-scatter
conda install -c pyg pytorch-cluster
conda install -c pyg pyg
pip install ogb
conda install -c dglteam/label/cu113 dgl
pip install yacs
pip install transformers
```


## 1. Download TAG datasets

### A. Original text attributes

| Dataset | Description |
| ----- |  ---- |
| ogbn-arxiv  | The [OGB](https://ogb.stanford.edu/docs/nodeprop/) provides the mapping from MAG paper IDs into the raw texts of titles and abstracts. <br/>Download the dataset [here](https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz), unzip and move it to `dataset/ogbn_arxiv_orig`. The dataset size is 200M.|
|Cora| Download the dataset [here](https://drive.google.com/drive/folders/1qRlKEuxjMJwatHtO2cIYbyVPpCesG4lf?usp=sharing) and move it to `dataset/cora_orig`. The dataset size is 2.6G.|
PubMed | Download the dataset [here](https://drive.google.com/drive/folders/1Wi-9isAxXZ62XkBzTlOhclUbW94vriGr?usp=sharing) and move it to `dataset/PubMed_orig`. The dataset size is 115M.|


### B. LLM responses
| Dataset | Description |
| ----- |  ---- |
| ogbn-arxiv  | Download the dataset [here](https://drive.google.com/drive/folders/1ZO3r6Ek_FJHFEmDX9LeuICy2kT6zX73d?usp=sharing) and move it to `gpt_responses/ogbn_arxiv`. The dataset size is 662M.|
|Cora| Download the dataset [here](https://drive.google.com/drive/folders/1GZnuf22Q7nchvNiOslq4PmM3E-1dpCgi?usp=sharing) and move it to `gpt_responses/cora`. The dataset size is 11M.|
PubMed | Download the dataset [here](https://drive.google.com/drive/folders/1YYuy72om88Pch7YbMLBHUyH4vEseED8B?usp=sharing) and move it to `gpt_responses/PubMed`. The dataset size is 77M.|


## 2. Fine-tuning the LMs
### To use the orginal text attributes
```
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM dataset ogbn-arxiv
```

### To use the GPT responses
```
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM dataset ogbn-arxiv use_gpt True
```


## 3. Training the GNNs

### To use different GNN models
```
python -m core.trainEnsemble gnn.model.name GCN
python -m core.trainEnsemble gnn.model.name SAGE
python -m core.trainEnsemble gnn.model.name RevGAT gnn.train.use_dgl True gnn.train.lr 0.002 gnn.train.dropout 0.75
```

### To use different types of features
```
# Our enriched features
python -m core.trainEnsemble gnn.train.feature_type TA_P_E

# Our individual features
python -m core.trainGNN gnn.train.feature_type TA
python -m core.trainGNN gnn.train.feature_type E
python -m core.trainGNN gnn.train.feature_type P

# OGB features
python -m core.trainGNN gnn.train.feature_type ogb
```

## Reproducibility
Use `run.sh` to run the codes and reproduce the published results.