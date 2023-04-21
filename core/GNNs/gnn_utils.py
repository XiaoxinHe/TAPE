import numpy as np

import dgl
import torch
from torch.utils.data import Dataset as TorchDataset


class CustomDGLDataset(TorchDataset):
    def __init__(self, pyg_dataset):
        self.pyg_dataset = pyg_dataset

    def __len__(self):
        return len(self.pyg_dataset)

    def __getitem__(self, idx):
        data = self.pyg_dataset[idx]
        g = dgl.DGLGraph()
        g.add_nodes(data.num_nodes)
        g.add_edges(data.edge_index[0], data.edge_index[1])
        g.ndata['feat'] = torch.FloatTensor(data.x)
        g.ndata['label'] = torch.LongTensor(data.y)
        if data.edge_attr is not None:
            g.edata['feat'] = torch.FloatTensor(data.edge_attr)

        return g

    @property
    def train_mask(self):
        return self.pyg_dataset.train_mask

    @property
    def val_mask(self):
        return self.pyg_dataset.val_mask

    @property
    def test_mask(self):
        return self.pyg_dataset.test_mask


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

    dataset, _ = get_raw_text(False)

    if use_dgl:
        dataset = CustomDGLDataset(dataset)

    return dataset


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
