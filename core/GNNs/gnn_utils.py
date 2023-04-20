from sklearn.metrics import roc_auc_score
import pandas as pd
import os
import numpy as np


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


def load_ogb_graph_structure_only(dataset):
    from ogb.nodeproppred import DglNodePropPredDataset
    data = DglNodePropPredDataset(dataset, root='dataset')
    g, labels = data[0]
    split_idx = data.get_idx_split()
    labels = labels.squeeze().numpy()
    return g, labels, split_idx


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
