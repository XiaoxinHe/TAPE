import time
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset
from core.data_utils.load_cora import get_cora_casestudy
from core.data_utils.load_pubmed import get_pubmed_casestudy
from core.data_utils.load_citeseer import get_citeseer_casestudy
from transformers import BertTokenizer
import torch
import json
import pandas as pd
import torch_geometric.transforms as T


def get_raw_text_cora():
    data, data_citeid = get_cora_casestudy()
    with open('dataset/Cora-Orig/mccallum/cora/papers')as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        pid_filename[pid] = fn

    path = 'dataset/Cora-Orig/mccallum/cora/extractions/'
    text = []
    for pid in tqdm(data_citeid):
        fn = pid_filename[pid]
        with open(path+fn) as f:
            lines = f.read().splitlines()

        for line in lines:
            if 'Title:' in line:
                ti = line
            if 'Abstract:' in line:
                ab = line
        text.append(ti+'\n'+ab)
    return data, text


def get_raw_text_pubmed():
    data, data_pubid = get_pubmed_casestudy()
    f = open('dataset/Pubmed-Diabetes/pubmed.json')
    pubmed = json.load(f)
    df_pubmed = pd.DataFrame.from_dict(pubmed)

    AB = df_pubmed['AB'].fillna("")
    TI = df_pubmed['TI'].fillna("")
    text = []
    for ti, ab in zip(TI, AB):
        t = 'Title: ' + ti + '\n'+'Abstract: ' + ab
        text.append(t)
    return data, text


def get_raw_text_citeseer():
    data, data_citeid = get_citeseer_casestudy()
    with open('dataset/CiteSeer-Orig/citeseer_texts.txt') as f:
        lines = f.read().splitlines()
    paper_ids = [lines[i] for i in range(len(lines)) if i % 3 == 0]
    abstracts = [lines[i] for i in range(len(lines)) if i % 3 == 1]
    # labels = [lines[i] for i in range(len(lines)) if i % 3 == 2]
    pid_ab = {}
    for i, j in zip(paper_ids, abstracts):
        pid_ab[i] = j
    text = []
    for pid in data_citeid:
        if pid in pid_ab:
            text.append(pid_ab[pid])
        else:
            text.append("")
    return data, text


def get_raw_text_arxiv():
    dataset = PygNodePropPredDataset(
        'ogbn-arxiv', transform=T.ToSparseTensor())
    data = dataset[0]
    idx_splits = dataset.get_idx_split()

    train_mask = torch.zeros(data.x.size(0)).bool()
    val_mask = torch.zeros(data.x.size(0)).bool()
    test_mask = torch.zeros(data.x.size(0)).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.edge_index = data.adj_t.to_symmetric()

    nodeidx2paperid = pd.read_csv(
        'dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')

    raw_text = pd.read_csv('dataset/ogbn_arxiv/titleabs.tsv',
                           sep='\t', header=None, names=['paper id', 'title', 'abs'])
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')
    text = []
    for ti, ab in zip(df['title'], df['abs']):
        t = 'Title: ' + ti + '\n' + 'Abstract: ' + ab
        text.append(t)
    return data, text


def _preprocess(input_text, tokenizer):
    '''
    Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
    '''
    return tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=128,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )


def preprocessing(dataset):

    print("[!] Preprocessing")
    start = time.time()
    if dataset == 'cora':
        data, text = get_raw_text_cora()
    elif dataset == 'pubmed':
        data, text = get_raw_text_pubmed()
    elif dataset == 'citeseer':
        data, text = get_raw_text_citeseer()
    elif dataset == 'ogbn-arxiv':
        data, text = get_raw_text_arxiv()

    token_id = []
    attention_masks = []
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case=True)
    for sample in text:
        encoding_dict = _preprocess(sample, tokenizer)
        token_id.append(encoding_dict['input_ids'])
        attention_masks.append(encoding_dict['attention_mask'])
    token_id = torch.cat(token_id, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    print("Time: ", time.time()-start)

    return data, token_id, attention_masks


def generate_node_embedding(model, dataloader, device):
    print("[!] Generating node embeddings")
    start = time.time()
    model.to(device)
    features = []
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        output = model(batch)
        features.append(output.detach().cpu())
    features = torch.cat(features, dim=0)
    print("Time: ", time.time()-start)
    return features
