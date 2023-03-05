import numpy as np
import torch
from time import time

from core.GNNs.GCN.model import GCN
from core.utils.modules.early_stopper import EarlyStopping
from core.utils.function.os_utils import init_path
from core.utils.function.np_utils import save_memmap
from core.GNNs.kd_gnn_trainer import load_data

early_stop = 50
feat_shrink = ""


class Z(torch.nn.Module):
    def __init__(self, z):
        super(Z, self).__init__()
        self.Z = torch.nn.Parameter(z)

    def forward(self):
        return self.Z


class ZTrainer():
    def __init__(self, args):
        self.device = args.device
        self.stage = args.stage
        self.dataset = args.dataset
        self.penalty = args.penalty
        self.lr = args.lr
        self.gnn_num_layers = args.gnn_num_layers
        self.gnn_dropout = args.gnn_dropout

        self.dim = feat_shrink if feat_shrink else 768
        self.epochs = 1000
        self.ckpt = init_path(f"output/{self.dataset}/z.emb{self.stage}")
        self.model_ckpt = init_path(f"output/{self.dataset}/z{self.stage}.pt")

        # ! Load data
        data = load_data(self.dataset)
        self.data = data.to(self.device)
        self.n_nodes = self.data.x.size(0)

    def _train(self):
        # ! Shared
        self.model.train()
        self.gnn.eval()
        self.optimizer.zero_grad()

        # ! Specific
        z = self.model()
        logits = self.gnn(z, self.data.edge_index)  # small-graph
        train_acc = self.evaluator(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss_gnn = self.loss_func_gnn(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss0 = 0.5*self.penalty * \
            self.mse_loss(z, (self.lm_x-self.gamma/self.penalty))

        loss = loss_gnn + loss0
        loss.backward()
        self.optimizer.step()
        return loss.item(), loss_gnn.item(), loss0.item(),  train_acc

    @torch.no_grad()
    def _evaluate(self):
        self.gnn.eval()
        self.model.eval()
        z = self.model()
        logits = self.gnn(z, self.data.edge_index)  # small-graph
        val_acc = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc, logits

    def _load_data(self):
        lm_x = np.memmap(f"output/{self.dataset}/bert.emb{self.stage}", mode='r',
                         dtype=np.float32, shape=(self.n_nodes, self.dim))
        self.lm_x = torch.Tensor(np.array(lm_x)).to(self.device)

        gamma = np.memmap(f"output/{self.dataset}/gamma.emb{self.stage-1}", mode='r',
                          dtype=np.float32, shape=(self.n_nodes, self.dim))
        self.gamma = torch.Tensor(np.array(gamma)).to(self.device)

    def _load_gnn(self):
        gnn_ckpt = f"output/{self.dataset}/GNN{self.stage}.pt"
        self.gnn = GCN(in_channels=self.dim,
                       hidden_channels=self.dim,
                       out_channels=self.data.y.unique().size(0),
                       num_layers=self.gnn_num_layers,
                       dropout=self.gnn_dropout
                       ).to(self.device)
        self.gnn.load_state_dict(torch.load(gnn_ckpt))

    def _load_model_z(self):
        z = np.memmap(f"output/{self.dataset}/z.emb{self.stage-1}", mode='r',
                      dtype=np.float32, shape=(self.n_nodes, self.dim))
        z = torch.Tensor(np.array(z))
        self.model = Z(z.detach().clone()).to(self.device)

    def train(self):
        self._load_data()
        self._load_gnn()
        self._load_model_z()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.0)

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f'!!!!!Z Phase, trainable_params are {trainable_params}')
        self.stopper = EarlyStopping(
            patience=early_stop, path=self.model_ckpt) if early_stop > 0 else None
        self.loss_func_gnn = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()

        if 'ogbn' in self.dataset:
            from ogb.nodeproppred import Evaluator
            self.data.y = self.data.y.squeeze()
        else:
            from core.GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )["acc"]

        # ! Training
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            loss, loss_gnn, loss0, train_acc = self._train()
            val_acc, test_acc, _ = self._evaluate()
            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(
                        f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
            log_dict = {'Epoch': epoch, 'Loss': round(loss, 4),
                        'Loss(GNN)': round(loss_gnn, 4), 'Loss0': round(loss0, 8),
                        'TrainAcc': round(train_acc, 4), 'ValAcc': round(val_acc, 4), 'TestAcc': round(test_acc, 4),
                        'ES': es_str, 'GNN_epoch': epoch}
            print(log_dict)

        # ! Finished training, load checkpoints
        if self.stopper is not None:
            self.model.load_state_dict(torch.load(self.stopper.path))
        return self.model

    @torch.no_grad()
    def eval_and_save(self):
        val_acc, test_acc, logits = self._evaluate()
        res = {'z_val_acc': val_acc, 'z_test_acc': test_acc}
        print(res)

        z = self.model()
        save_memmap(z.cpu().numpy(), self.ckpt, dtype=np.float32)

    @torch.no_grad()
    def init(self):
        lm_x = np.memmap(f"output/{self.dataset}/bert.emb0",
                         mode='r',
                         dtype=np.float32,
                         shape=(self.n_nodes, feat_shrink if feat_shrink else 768))

        save_memmap(lm_x, self.ckpt, dtype=np.float32)
