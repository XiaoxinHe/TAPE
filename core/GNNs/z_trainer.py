import numpy as np
import torch
from time import time

from core.GNNs.GCN.model import GCN
from core.utils.modules.early_stopper import EarlyStopping
from core.preprocess import preprocessing
from core.utils.function.os_utils import init_path
from core.utils.function.np_utils import save_memmap


early_stop = 50

feat_shrink = 128


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
        self.penalty = 0.5
        self.epochs = 1000

        self.gnn_ckpt = f"output/{self.dataset}/GNN{self.stage}.pt"
        self.z_ckpt = f"output/{self.dataset}/z{self.stage}.pt"

        # ! Load data
        data = preprocessing(self.dataset, use_text=False)
        self.data = data.to(self.device)

        lm_x = np.memmap(f"output/{self.dataset}/bert.emb{self.stage}", mode='r',
                         dtype=np.float32, shape=(data.x.shape[0], feat_shrink))
        lm_x = torch.Tensor(np.array(lm_x))
        self.lm_x = lm_x.to(self.device)

        if self.stage == 0:
            self.model = Z(lm_x.detach().clone()).to(self.device)
        else:
            self.gnn = GCN(in_channels=self.lm_x.shape[1],
                           hidden_channels=128,
                           out_channels=data.y.unique().size(0),
                           num_layers=4,
                           dropout=0.0
                           ).to(self.device)
            self.gnn.load_state_dict(torch.load(self.gnn_ckpt))
            gamma = np.memmap(f"output/{self.dataset}/gamma.emb{self.stage-1}", mode='r',
                              dtype=np.float32, shape=(data.x.shape[0], feat_shrink))
            gamma = torch.Tensor(np.array(gamma))
            self.gamma = gamma.to(self.device)

            z = np.memmap(f"output/{self.dataset}/z.emb{self.stage-1}", mode='r',
                          dtype=np.float32, shape=(data.x.shape[0], feat_shrink))
            z = torch.Tensor(np.array(z))
            self.model = Z(z.detach().clone()).to(self.device)

            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=1e-2, weight_decay=0.0)

            trainable_params = sum(p.numel()
                                   for p in self.model.parameters() if p.requires_grad)

            print(f'!!!!!Z Phase, trainable_params are {trainable_params}')
            self.stopper = EarlyStopping(
                patience=early_stop, path=self.z_ckpt) if early_stop > 0 else None
            self.loss_func_gnn = torch.nn.CrossEntropyLoss()
            self.l2_loss = torch.nn.MSELoss()

            from core.GNNs.gnn_utils import Evaluator
            self._evaluator = Evaluator(name=self.dataset)
            self.evaluator = lambda pred, labels: self._evaluator.eval(
                {"y_pred": pred.argmax(dim=-1, keepdim=True),
                 "y_true": labels.view(-1, 1)}
            )["acc"]

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
            self.l2_loss(z, (self.lm_x-self.gamma/self.penalty))

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

    def train(self):
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
            log_dict = {'Epoch': epoch, 'Time': round(time() - t0, 4),
                        'Loss': round(loss, 4), 'Loss(GNN)': round(loss_gnn, 4), 'Loss0': round(loss0, 8),
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
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        print(res)
        z = self.model()
        save_memmap(z.cpu().numpy(), init_path(
            f"output/{self.dataset}/z.emb{self.stage}"), dtype=np.float32)

    @torch.no_grad()
    def save(self):
        z = self.model()
        save_memmap(z.cpu().numpy(), init_path(
            f"output/{self.dataset}/z.emb{self.stage}"), dtype=np.float32)
