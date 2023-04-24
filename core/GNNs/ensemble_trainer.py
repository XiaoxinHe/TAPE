from core.GNNs.gnn_utils import load_data
import torch
from time import time
from core.GNNs.GCN.model import GCN
from core.GNNs.SAGE.model import SAGE
from core.utils.modules.early_stopper import EarlyStopping
import numpy as np


LOG_FREQ = 10


class EnsembleTrainer():
    def __init__(self, args):
        self.device = args.device
        self.gnn_model_name = args.gnn_model_name
        self.lm_model_name = args.lm_model_name
        self.dataset_name = args.dataset_name
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.lr = args.lr
        self.combine = args.combine
        self.epochs = args.epochs
        self.combine = args.combine

        # ! Load data
        data = load_data(self.dataset_name)

        self.num_nodes = data.x.shape[0]
        self.num_classes = data.y.unique().size(0)

        data.y = data.y.squeeze()
        self.data = data.to(self.device)

        from core.GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset_name)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )["acc"]

    @ torch.no_grad()
    def _evaluate(self, logits):
        val_acc = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc

    @ torch.no_grad()
    def eval(self, logits):
        val_acc, test_acc = self._evaluate(logits)
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        print(res)
        return res
