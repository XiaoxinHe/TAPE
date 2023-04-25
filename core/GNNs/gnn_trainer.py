from core.GNNs.gnn_utils import load_data, load_gpt_preds
import torch
from time import time
from core.GNNs.GCN.model import GCN
from core.GNNs.SAGE.model import SAGE
from core.utils.modules.early_stopper import EarlyStopping
import numpy as np


LOG_FREQ = 10

lm_dim = {
    'microsoft/deberta-base': 768,
    'microsoft/deberta-large': 1024,
}


def _process(feature1, feature2, combine):
    print(f"!!! using {combine} feature")
    if combine == 'sum':
        return feature1 + feature2
    elif combine == 'prod':
        return feature1*feature2
    elif combine == 'concat':
        return torch.cat([feature1, feature2], dim=1)
    elif combine == 'f2':
        return feature2
    else:
        return feature1


class GNNTrainer():
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
        self.seed = args.seed

        # ! Load data
        data = load_data(self.dataset_name)

        self.num_nodes = data.x.shape[0]
        self.num_classes = data.y.unique().size(0)
        use_pred = self.combine == 'f3'

        # ! Init gnn feature
        topk = 3 if self.dataset_name == 'pubmed' else 5
        if args.use_ogb:
            print("Loading OGB features...")
            self.features = data.x.to(self.device)
        elif self.combine == 'f3':
            print("Loading top-k prediction features...")
            self.features = load_gpt_preds(
                self.dataset_name, topk).to(self.device)
        else:
            print("Loading pretrained LM features...")
            LM_emb_path = f"prt_lm/{self.dataset_name}-seed{self.seed}/{self.lm_model_name}.emb"
            LM_emb_path2 = f"prt_lm/{self.dataset_name}2-seed{self.seed}/{self.lm_model_name}.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            print(f"LM_emb_path2: {LM_emb_path2}")
            feature = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, lm_dim[self.lm_model_name])))
            ).to(torch.float32)
            feature2 = torch.from_numpy(np.array(
                np.memmap(
                    LM_emb_path2, mode='r',
                    dtype=np.float16,
                    shape=(self.num_nodes, lm_dim[self.lm_model_name])))
            ).to(torch.float32)

            self.features = _process(
                feature, feature2, self.combine).to(self.device)

        data.y = data.y.squeeze()
        self.data = data.to(self.device)
        # ! Trainer init
        if self.gnn_model_name == "GCN":
            self.model = GCN(in_channels=self.hidden_dim*topk if use_pred else self.features.shape[1],
                             hidden_channels=self.hidden_dim,
                             out_channels=self.num_classes,
                             num_layers=self.num_layers,
                             dropout=self.dropout,
                             use_pred=use_pred).to(self.device)

        elif self.gnn_model_name == "SAGE":
            self.model = SAGE(in_channels=self.hidden_dim*topk if use_pred else self.features.shape[1],
                              hidden_channels=self.hidden_dim,
                              out_channels=self.num_classes,
                              num_layers=self.num_layers,
                              dropout=self.dropout,
                              use_pred=use_pred).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.0)

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f'!!!!!GNN Phase, trainable_params are {trainable_params}')
        self.ckpt = f"output/{self.dataset_name}/{self.gnn_model_name}.pt"
        self.stopper = EarlyStopping(
            patience=args.early_stop, path=self.ckpt) if args.early_stop > 0 else None
        self.loss_func = torch.nn.CrossEntropyLoss()

        from core.GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset_name)
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
        self.optimizer.zero_grad()
        # ! Specific
        logits = self._forward(self.features, self.data.edge_index)
        loss = self.loss_func(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        # if self.dataset_name == 'ogbn-arxiv':
        #     src, dst = self.data.edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
        # else:
        #     src, dst = self.data.edge_index
        # pi = torch.softmax(logits[src], dim=-1)
        # pj = torch.softmax(logits[dst], dim=-1)
        # loss_tv = 10*(pi-pj).abs().sum() / (src.shape[0]*self.hidden_dim)
        # loss += loss_tv
        train_acc = self.evaluator(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()

        return loss.item(), train_acc

    @ torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        logits = self._forward(self.features, self.data.edge_index)
        val_acc = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc, logits

    def train(self):
        # ! Training
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            loss, train_acc = self._train()
            val_acc, test_acc, _ = self._evaluate()
            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(
                        f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
            if epoch % 10 == 0:
                log_dict = {'Epoch': epoch, 'Time': round(time() - t0, 4), 'Loss': round(loss, 4),
                            'TrainAcc': round(train_acc, 4), 'ValAcc': round(val_acc, 4), 'TestAcc': round(test_acc, 4),
                            'ES': es_str, 'GNN_epoch': epoch}
                print(log_dict)

        # ! Finished training, load checkpoints
        if self.stopper is not None:
            self.model.load_state_dict(torch.load(self.stopper.path))

        return self.model

    @ torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model.state_dict(), self.ckpt)
        val_acc, test_acc, logits = self._evaluate()
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        print(res)
        return logits, res
