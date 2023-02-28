import numpy as np
import torch
from time import time
from core.GNNs.GCN.model import KDGCN
from core.utils.modules.early_stopper import EarlyStopping
from core.preprocess import preprocessing
from core.utils.function.os_utils import init_path
from core.utils.function.np_utils import save_memmap


early_stop = 50


class GNNTrainer():
    def __init__(self, args):
        self.device = args.device
        self.stage = args.stage
        self.dataset = args.dataset
        self.epochs = 200
        self.pred = init_path(f"output/{self.dataset}/gnn.pred{self.stage}")
        self.emb = init_path(f"output/{self.dataset}/gnn.emb{self.stage}")

        # ! Load data
        data = preprocessing(self.dataset, use_text=False)

        # ! Init gnn feature
        emb = np.memmap(f'output/{self.dataset}/bert.emb{self.stage}',
                        mode='r', dtype=np.float32, shape=(data.x.shape[0], 128))
        emb = torch.Tensor(np.array(emb))
        self.features = emb.to(self.device)
        # self.features = data.x.to(self.device)
        self.data = data.to(self.device)
        self.n_labels = self.data.y.unique().size(0)

        # ! Trainer init
        self.model = KDGCN(in_channels=self.features.shape[1],
                           hidden_channels=128,
                           out_channels=self.n_labels,
                           num_layers=4,
                           dropout=0.0).to(self.device)
        # if self.stage > 0:
        #     self.model.load_state_dict(torch.load(
        #         f"output/{self.dataset}/GNN{self.stage-1}.pt"))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=0.0)

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f'!!!!!GNN Phase, trainable_params are {trainable_params}')
        self.ckpt = f"output/{self.dataset}/GNN{self.stage}.pt"
        self.stopper = EarlyStopping(
            patience=early_stop, path=self.ckpt) if early_stop > 0 else None
        self.loss_func = torch.nn.CrossEntropyLoss()

        from core.GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )["acc"]

    def _train(self):
        self.model.train()
        self.optimizer.zero_grad()
        embs, logits = self.model(self.features, self.data.edge_index)
        loss = self.loss_func(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        train_acc = self.evaluator(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()

        return loss.item(), train_acc

    @ torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        embs, logits = self.model(self.features, self.data.edge_index)
        val_acc = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc, embs, logits

    def train(self):
        # ! Training
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            loss, train_acc = self._train()
            val_acc, test_acc, _, _ = self._evaluate()
            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(
                        f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
            log_dict = {'Epoch': epoch, 'Time': round(time() - t0, 4), 'Loss': round(loss, 4), 'TrainAcc': round(train_acc, 4), 'ValAcc': round(val_acc, 4), 'TestAcc': round(test_acc, 4),
                        'ES': es_str, 'GNN_epoch': epoch}
            print(log_dict)

        # ! Finished training, load checkpoints
        if self.stopper is not None:
            self.model.load_state_dict(torch.load(self.stopper.path))

        return self.model

    @ torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model.state_dict(), self.ckpt)
        val_acc, test_acc, embs, logits = self._evaluate()
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        print(res)
        save_memmap(logits.cpu().numpy(), self.pred, dtype=np.float32)
        save_memmap(embs.cpu().numpy(), self.emb, dtype=np.float32)
