import numpy as np

import dgl
import torch
from torch.utils.data import Dataset as TorchDataset
import csv


def to_unique_list(my_list):
    unique_list = []
    seen = set()
    for item in my_list:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    return unique_list


# def _modify(my_list, train_mask, label):
#     label = label.squeeze().tolist()
#     for i, row in enumerate(my_list):
#         if train_mask[i]:
#             my_list[i].insert(0, label[i])
#     return my_list


def _load(dataset):
    loaded_list = []
    with open(f'gpt_preds/{dataset}.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            inner_list = []
            for value in row:
                inner_list.append(int(value))
            loaded_list.append(inner_list)
    return loaded_list


def load_gpt_preds(dataset, topk):
    preds = _load(dataset)
    # preds = _modify(preds, train_mask, label)
    # preds = [to_unique_list(i) for i in preds]

    pl = torch.zeros(len(preds), topk, dtype=torch.long)
    for i, pred in enumerate(preds):
        pl[i][:len(pred)] = torch.tensor(pred[:topk], dtype=torch.long)+1
    return pl


class CustomDGLDataset(TorchDataset):
    def __init__(self, name, pyg_data):
        self.name = name
        self.pyg_data = pyg_data

    def __len__(self):
        return 1

    # def __getitem__(self, idx):
    #     if self.name == 'ogbn-arxiv':
    #         from ogb.nodeproppred import DglNodePropPredDataset
    #         dgl_dataset = DglNodePropPredDataset(name='ogbn-arxiv')
    #         g, labels = dgl_dataset[0]
    #         feat = g.ndata['feat']
    #         g = dgl.to_bidirected(g)
    #         print(
    #             f"Using GAT based methods,total edges before adding self-loop {g.number_of_edges()}")
    #         g = g.remove_self_loop().add_self_loop()
    #         print(f"Total edges after adding self-loop {g.number_of_edges()}")
    #         g.ndata['feat'] = feat
    #         g.ndata['label'] = labels.squeeze()
    #     else:
    #         data = self.pyg_data
    #         g = dgl.DGLGraph()
    #         g.add_nodes(data.num_nodes)
    #         g.add_edges(data.edge_index[0], data.edge_index[1])
    #         g.ndata['feat'] = torch.FloatTensor(data.x)
    #         g.ndata['label'] = torch.LongTensor(data.y)
    #         if data.edge_attr is not None:
    #             g.edata['feat'] = torch.FloatTensor(data.edge_attr)
    #     return g

    def __getitem__(self, idx):

        data = self.pyg_data
        g = dgl.DGLGraph()
        if self.name == 'ogbn-arxiv':
            edge_index = data.edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
        else:
            edge_index = data.edge_index
        g.add_nodes(data.num_nodes)
        g.add_edges(edge_index[0], edge_index[1])

        if data.edge_attr is not None:
            g.edata['feat'] = torch.FloatTensor(data.edge_attr)
        if self.name == 'ogbn-arxiv':
            g = dgl.to_bidirected(g)
            print(
                f"Using GAT based methods,total edges before adding self-loop {g.number_of_edges()}")
            g = g.remove_self_loop().add_self_loop()
            print(f"Total edges after adding self-loop {g.number_of_edges()}")
        g.ndata['feat'] = torch.FloatTensor(data.x)
        g.ndata['label'] = torch.LongTensor(data.y)
        return g

    @property
    def train_mask(self):
        return self.pyg_data.train_mask

    @property
    def val_mask(self):
        return self.pyg_data.val_mask

    @property
    def test_mask(self):
        return self.pyg_data.test_mask


def load_data(dataset, use_dgl=False):
    if dataset == 'cora':
        from core.data_utils.load_cora import get_raw_text_cora as get_raw_text
    elif dataset == 'pubmed':
        from core.data_utils.load_pubmed import get_raw_text_pubmed as get_raw_text
    elif dataset == 'citeseer':
        from core.data_utils.load_citeseer import get_raw_text_citeseer as get_raw_text
    elif dataset == 'ogbn-arxiv':
        from core.data_utils.load_arxiv import get_raw_text_arxiv as get_raw_text
    elif dataset == 'ogbn-products':
        from core.data_utils.load_products import get_raw_text_products as get_raw_text

    data, _ = get_raw_text(False)

    if use_dgl:
        data = CustomDGLDataset(dataset, data)

    return data


def get_gnn_trainer(model):
    if model in ['GCN', 'RevGAT', 'SAGE']:
        from core.GNNs.gnn_trainer import GNNTrainer
    # elif model in ['SAGE']:
    #     from models.GNNs.minibatch_trainer import BatchGNNTrainer as GNNTrainer
    # elif model in ['SAGN']:
    #     from models.GNNs.SAGNTrainer import SAGN_Trainer as GNNTrainer
    # elif model in ['EnGCN']:
    #     from models.GNNs.EnGCNTrainer import EnGCNTrainer as GNNTrainer
    # elif model in ['GAMLP']:
    #     from models.GNNs.GAMLPTrainer import GAMLP_Trainer as GNNTrainer
    # elif model in ['GAMLP_DDP']:
    #     from models.GNNs.GAMLP_DDP_Trainer import GAMLP_DDP_Trainer as GNNTrainer
    else:
        raise ValueError(f'GNN-Trainer for model {model} is not defined')
    return GNNTrainer


class Evaluator:
    def __init__(self, name):
        self.name = name

    def eval(self, input_dict):
        y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return {'acc': sum(acc_list)/len(acc_list)}


def compute_loss(logits, labels, loss_func):
    loss = loss_func(logits, labels)
    return loss
