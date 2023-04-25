from core.GNNs.gnn_utils import load_data, load_gpt_preds
import torch
from time import time
from core.GNNs.RevGAT.model import RevGAT
from core.utils.modules.early_stopper import EarlyStopping
import numpy as np


LOG_FREQ = 10

lm_dim = {
    'microsoft/deberta-base': 768,
    'microsoft/deberta-large': 1024,
}


def _process(feature1, feature2, combine):
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


class DGLGNNTrainer():
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
        self.weight_decay = args.weight_decay

        self.n_heads = 3
        self.input_drop = 0.25
        self.attn_drop = 0.0
        self.edge_drop = 0.3
        self.no_attn_dst = True
        self.use_norm = False
        self.group = 2
        self.input_norm = 'T'

        # ! Load data
        dataset = load_data(self.dataset_name, use_dgl=True)
        data = dataset[0]

        self.train_mask = dataset.train_mask
        self.val_mask = dataset.val_mask
        self.test_mask = dataset.test_mask
        self.y = data.ndata['label'].squeeze().to(self.device)

        self.num_nodes = data.num_nodes()
        self.num_classes = self.y.unique().size(0)

        # ! Init gnn feature
        if args.use_ogb:
            print("Loading OGB features...")
            self.features = data.ndata['feat'].to(self.device)
        elif self.combine == 'f3':
            print("Loading top-k prediction features...")
            self.features = load_gpt_preds(
                self.dataset_name, 5).to(self.device)
        else:
            print("Loading pretrained LM features...")
            LM_emb_path = f"prt_lm/{self.dataset_name}/{self.lm_model_name}.emb"
            LM_emb_path2 = f"prt_lm/{self.dataset_name}2/{self.lm_model_name}.emb"
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

            # self.preds = torch.load('gpt_labels.pt').to(self.device)

        self.data = data.to(self.device)
        # ! Trainer init
        use_pred = self.combine == 'f3'
        if self.gnn_model_name == "RevGAT":
            self.model = RevGAT(in_feats=self.hidden_dim*5 if use_pred else self.features.shape[1],
                                n_classes=self.num_classes,
                                n_hidden=self.hidden_dim,
                                n_layers=self.num_layers,
                                n_heads=self.n_heads,
                                activation=torch.nn.Mish(),
                                dropout=self.dropout,
                                input_drop=self.input_drop,
                                attn_drop=self.attn_drop,
                                edge_drop=self.edge_drop,
                                use_attn_dst=not self.no_attn_dst,
                                use_symmetric_norm=self.use_norm,
                                group=self.group,
                                input_norm=self.input_norm == 'T',
                                use_pred=use_pred
                                ).to(self.device)

        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f'!!!!!GNN Phase, trainable_params are {trainable_params}')
        self.ckpt = f"output/{self.dataset_name}/{self.gnn_model_name}.pt"
        self.stopper = EarlyStopping(
            patience=args.early_stop, path=self.ckpt) if args.early_stop > 0 else None
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

        from core.GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset_name)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )["acc"]

    def _forward(self, *args):
        logits = self.model(*args)  # small-graph
        return logits

    def _train(self):
        # ! Shared
        self.model.train()
        self.optimizer.zero_grad()
        # ! Specific
        logits = self._forward(self.data, self.features)
        loss = self.loss_func(
            logits[self.train_mask], self.y[self.train_mask])
        train_acc = self.evaluator(
            logits[self.train_mask], self.y[self.train_mask])
        loss.backward()
        self.optimizer.step()

        return loss.item(), train_acc

    @ torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        logits = self._forward(self.data, self.features)
        val_acc = self.evaluator(
            logits[self.val_mask], self.y[self.val_mask])
        test_acc = self.evaluator(
            logits[self.test_mask], self.y[self.test_mask])
        return val_acc, test_acc, logits

    def train(self):
        # ! Training
        for epoch in range(self.epochs):
            if epoch <= 50 and self.gnn_model_name == 'RevGAT':
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr * epoch / 50
            t0, es_str = time(), ''
            loss, train_acc = self._train()
            val_acc, test_acc, _ = self._evaluate()
            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(
                        f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
            if epoch % LOG_FREQ == 0:
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
        val_acc, test_acc, logits = self._evaluate()
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        print(res)
        return logits, res
