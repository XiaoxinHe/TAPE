import numpy as np
import torch
from time import time

from core.GNNs.GCN.model import GCN
from core.utils.modules.early_stopper import EarlyStopping
from core.preprocess import preprocessing
from core.utils.function.os_utils import init_path
from core.utils.function.np_utils import save_memmap

DATASET = 'cora'
LOG_FREQ = 10
checkpoint_file = ['output/GNN.pt', 'output/Z.pt']
early_stop = 50


class Z(torch.nn.Module):
    def __init__(self, z):
        super(Z, self).__init__()
        self.Z = torch.nn.Parameter(z)

    def forward(self):
        return self.Z


class GNNTrainer():
    def __init__(self, device):
        self.device = device
        self.epochs = 200

        # ! Load data
        data = preprocessing(DATASET, use_text=False)

        # ! Init gnn feature
        lm_x = np.memmap('output/bert.emb', mode='r',
                         dtype=np.float16, shape=(data.x.shape[0], 768))
        lm_x = torch.Tensor(np.array(lm_x))

        self.lm_x = lm_x.to(device)
        self.data = data.to(device)

        # ! Trainer init
        self.model = GCN(in_channels=self.lm_x.shape[1],
                         hidden_channels=128,
                         out_channels=data.y.unique().size(0),
                         num_layers=4,
                         dropout=0.0).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=0.0)

        self.model_z = Z(lm_x.detach().clone()).to(self.device)
        self.optimizer_z = torch.optim.Adam(
            self.model_z.parameters(), lr=0.001, weight_decay=0.0)

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f'!!!!!GNN Phase, trainable_params are {trainable_params}')
        self.stopper = EarlyStopping(
            patience=early_stop, path=checkpoint_file) if early_stop > 0 else None
        self.loss_func_gnn = torch.nn.CrossEntropyLoss()
        self.loss_func_z = torch.nn.CosineSimilarity()

        from core.GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=DATASET)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )["acc"]

    def _forward(self, x, edge_index):
        logits = self.model(x, edge_index)  # small-graph
        return logits

    def _train(self):
        # ! Shared
        self.model.train()
        self.model_z.train()
        self.optimizer.zero_grad()
        self.optimizer_z.zero_grad()

        # ! Specific
        z = self.model_z()
        logits = self._forward(z, self.data.edge_index)
        loss_gnn = self.loss_func_gnn(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss_z = (1-self.loss_func_z(z, self.lm_x)).mean()
        loss = loss_gnn + loss_z
        train_acc = self.evaluator(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        self.optimizer_z.step()
        return loss.item(), loss_gnn.item(), loss_z.item(), train_acc

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        self.model_z.eval()
        z = self.model_z()
        logits = self._forward(z, self.data.edge_index)
        val_acc = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc, logits

    def train(self):
        # ! Training
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            loss, loss_gnn, loss_z, train_acc = self._train()
            val_acc, test_acc, _ = self._evaluate()
            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(
                    val_acc, [self.model, self.model_z], epoch)
                if es_flag:
                    print(
                        f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
            log_dict = {'Epoch': epoch, 'Time': round(time() - t0, 4),
                        'Loss': round(loss, 4), 'Loss(GNN)': round(loss_gnn, 4), 'Loss(Z)': round(loss_z, 8),
                        'TrainAcc': round(train_acc, 4), 'ValAcc': round(val_acc, 4), 'TestAcc': round(test_acc, 4),
                        'ES': es_str, 'GNN_epoch': epoch}
            print(log_dict)

        # ! Finished training, load checkpoints
        if self.stopper is not None:
            self.model.load_state_dict(torch.load(self.stopper.path[0]))
            self.model_z.load_state_dict(torch.load(self.stopper.path[1]))
        return self.model, self.model_z

    @torch.no_grad()
    def eval_and_save(self):
        val_acc, test_acc, logits = self._evaluate()
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        print(res)
        z = self.model_z()
        save_memmap(z.cpu().numpy(), init_path(
            "output/z.emb"), dtype=np.float16)
