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
    data = torch.load('dataset/ogbn_products/ogbn-products_subset.pt')
    text = pd.read_csv('dataset/ogbn_products_orig/ogbn-products_subset.csv')
    text = [f'Product:{ti}; Description: {cont}\n'for ti,
            cont in zip(text['title'], text['content'])]

    data.edge_index = data.adj_t.to_symmetric()

    if not use_text:
        return data, None

    return data, text


if __name__ == '__main__':
    data, text = get_raw_text_products(True)
    print(data)
    print(text[0])
