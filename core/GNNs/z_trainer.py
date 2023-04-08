import torch_geometric
from scipy import sparse as sp

from core.GNNs.kd_gnn_trainer import load_data
import numpy as np
import torch
from time import time

from core.GNNs.GCN.model import GCN
from core.utils.modules.early_stopper import EarlyStopping
from core.utils.function.os_utils import init_path
from core.utils.function.np_utils import save_memmap


early_stop = 50
LOG_FREQ = 10
feat_shrink = ""


class Z(torch.nn.Module):
    def __init__(self, z):
        super(Z, self).__init__()
        self.Z = torch.nn.Parameter(z)

    def forward(self):
        return self.Z


def lagrangien(g):
    degree = torch_geometric.utils.degree(g.edge_index[0], g.num_nodes)
    A = torch_geometric.utils.to_scipy_sparse_matrix(
        g.edge_index, num_nodes=g.num_nodes)
    N = sp.diags(np.array(degree.clip(1) ** -0.5, dtype=float))
    L = sp.eye(g.num_nodes) - N * A * N
    return torch.Tensor(L.todense())


class ZTrainer():
    def __init__(self, args):
        self.device = args.device
        self.stage = args.stage
        self.dataset = args.dataset
        self.epochs = 1000
        self.lr = args.lr

        self.dim = feat_shrink if feat_shrink else 768
        self.gnn_num_layers = args.gnn_num_layers
        self.penalty = args.penalty
        self.gamma = args.gamma

        self.emb = f"output/{self.dataset}/z.emb"
        self.ckpt = init_path(f"output/{self.dataset}/z.pt")

        # ! Load data
        data = load_data(self.dataset)
        self.L = lagrangien(data).to(self.device)
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
        loss_cons = 0.5*self.penalty * \
            self.mse_loss(z, (self.lm_x-self.gamma/self.penalty))
        dig_loss = self.gamma*torch.trace(z.T @ self.L @ z)/self.dim
        loss = loss_gnn + loss_cons + dig_loss
        loss.backward()
        self.optimizer.step()
        return loss_gnn.item(), loss_cons.item(), dig_loss.item(), train_acc

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
        lm_x = np.memmap(f"output/{self.dataset}/bert.emb", mode='r',
                         dtype=np.float32, shape=(self.n_nodes, self.dim))
        self.lm_x = torch.Tensor(np.array(lm_x)).to(self.device)

        gamma = np.memmap(f"output/{self.dataset}/gamma.emb", mode='r',
                          dtype=np.float32, shape=(self.n_nodes, self.dim))
        self.gamma = torch.Tensor(np.array(gamma)).to(self.device)

    def _load_gnn(self):
        gnn_ckpt = f"output/{self.dataset}/GNN.pt"
        self.gnn = GCN(in_channels=self.dim,
                       hidden_channels=self.dim,
                       out_channels=self.data.y.unique().size(0),
                       num_layers=self.gnn_num_layers,
                       ).to(self.device)
        self.gnn.load_state_dict(torch.load(gnn_ckpt))

    def _load_model_z(self):
        z = np.memmap(init_path(self.emb), mode='r', dtype=np.float32,
                      shape=(self.n_nodes, self.dim))
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
            patience=early_stop, path=self.ckpt) if early_stop > 0 else None
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
            loss_gnn, loss_cons, loss_dir, train_acc = self._train()
            val_acc, test_acc, _ = self._evaluate()
            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(
                        f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
            if epoch % LOG_FREQ == 0:
                log_dict = {'Epoch': epoch,
                            'Loss(GNN)': round(loss_gnn, 4), 'Loss(Cons)': round(loss_cons, 4), 'Loss(Dir)': round(loss_dir, 4),
                            'TrainAcc': round(train_acc, 4), 'ValAcc': round(val_acc, 4), 'TestAcc': round(test_acc, 4),
                            'ES': es_str}
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
        save_memmap(z.cpu().numpy(), init_path(self.emb), dtype=np.float32)

    @torch.no_grad()
    def init(self):
        lm_x = np.memmap(f"prt_lm/{self.dataset}/bert.emb",
                         mode='r',
                         dtype=np.float32,
                         shape=(self.n_nodes, feat_shrink if feat_shrink else 768))

        save_memmap(lm_x, init_path(self.emb), dtype=np.float32)
