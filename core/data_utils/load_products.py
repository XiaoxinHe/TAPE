from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import json
import numpy as np
import os
import time
from core.utils import time_logger

FILE = 'dataset/ogbn_products_orig/ogbn-products.csv'


@time_logger
def _process():
    if os.path.isfile(FILE):
        return

    print("Processing raw text...")
    ts = time.time()

    data = []
    files = ['dataset/ogbn_products/Amazon-3M.raw/trn.json',
             'dataset/ogbn_products/Amazon-3M.raw/tst.json']
    for file in files:
        with open(file) as f:
            for line in f:
                data.append(json.loads(line))

    df = pd.DataFrame(data)
    df.set_index('uid', inplace=True)

    nodeidx2asin = pd.read_csv(
        'dataset/ogbn_products/mapping/nodeidx2asin.csv.gz', compression='gzip')

    dataset = PygNodePropPredDataset(
        name='ogbn-products', transform=T.ToSparseTensor())
    graph = dataset[0]
    graph.n_id = np.arange(graph.num_nodes)
    graph.n_asin = nodeidx2asin.loc[graph.n_id]['asin'].values

    graph_df = df.loc[graph.n_asin]
    graph_df['nid'] = graph.n_id
    graph_df.reset_index(inplace=True)

    if not os.path.isdir('dataset/ogbn_products_orig'):
        os.mkdir('dataset/ogbn_products_orig')
    pd.DataFrame.to_csv(graph_df, FILE,
                        index=False, columns=['uid', 'nid', 'title', 'content'])


def get_raw_text_products(use_text=False, seed=0):
    dataset = PygNodePropPredDataset(
        name='ogbn-products', transform=T.ToSparseTensor())
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.edge_index = data.adj_t.to_symmetric()

    if not use_text:
        return data, None

    _process()
    with open(FILE) as f:
        df = pd.read_csv(f)
    df['title'].fillna("", inplace=True)
    df['content'].fillna("", inplace=True)
    text = []
    for ti, ab in zip(df['title'], df['content']):
        t = 'Title: ' + ti.strip() + '\n' + 'Content: ' + ab.strip()
        text.append(t)

    return data, text
