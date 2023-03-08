import numpy as np
import torch
from time import time
from core.GNNs.GCN.model import KDGCN
from core.utils.modules.early_stopper import EarlyStopping
from core.preprocess import preprocessing
from core.utils.function.os_utils import init_path
from core.utils.function.np_utils import save_memmap


early_stop = 50
feat_shrink = ""

class GNNTrainer():
    def __init__(self, args):
        self.device = args.device
        self.stage = args.stage
        self.dataset = args.dataset
        self.epochs = 200
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.dim = feat_shrink if feat_shrink else 768
        self.pred = init_path(f"output/{self.dataset}/gnn.pred{self.stage}")
        self.emb = init_path(f"output/{self.dataset}/gnn.emb{self.stage}")

        # ! Load data
        data = preprocessing(self.dataset, use_text=False)

        # ! Init gnn feature
        emb = np.memmap(f'output/{self.dataset}/bert.emb{self.stage}',
                        mode='r', dtype=np.float32, shape=(data.x.shape[0], 768))
        emb = torch.Tensor(np.array(emb))
        self.features = emb.to(self.device)

        # self.features = data.x.to(self.device)
        self.data = data

        self.n_nodes = self.data.x.size(0)
        self.n_labels = self.data.y.unique().size(0)
        self._prune_graph()
        self.data = data.to(self.device)
        # ! Trainer init
        self.model = KDGCN(in_channels=self.features.shape[1],
                           hidden_channels=self.dim,
                           out_channels=self.n_labels,
                           num_layers=self.num_layers,
                           dropout=self.dropout).to(self.device)
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

    # def _prune_graph(self):
    #     print("pruning graph")
    #     print(self.data)
    #     gnn_emb = np.memmap(f'output/cora/gnn.emb0', mode='r',
    #                         dtype=np.float32, shape=(self.n_nodes, 128))
    #     gnn_emb = torch.Tensor(np.array(gnn_emb))
    #     gnn_emb = gnn_emb/torch.norm(gnn_emb, dim=-1, keepdim=True)
    #     gnn_sim = torch.matmul(gnn_emb, gnn_emb.T)

    #     emb = np.memmap(f'output/cora/bert.emb0', mode='r',
    #                     dtype=np.float32, shape=(self.n_nodes, 128))
    #     emb = torch.Tensor(np.array(emb))
    #     emb = emb/torch.norm(emb, dim=-1, keepdim=True)
    #     sim = torch.matmul(emb, emb.T)
        
    #     adj = torch.zeros(self.n_nodes, self.n_nodes).bool()
    #     adj[self.data.edge_index[0], self.data.edge_index[1]] = True

    #     _, indices = torch.topk(
    #         (gnn_sim-sim).view(-1), k=int(self.n_nodes**2 * 0.01))
    #     row, col = indices//self.n_nodes, indices % self.n_nodes
    #     adj[row, col] = False

    #     # _, indices = torch.topk(
    #     #     (sim-gnn_sim).view(-1), k=int(self.data.edge_index.size(1) * 0.2))
    #     # row, col = indices//self.n_nodes, indices % self.n_nodes
    #     # adj[row, col] = True

    #     self.data.edge_index = adj.nonzero().T
    #     print(self.data)
    
    
    def _prune_graph(self):
        src, dst = self.data.edge_index
        adj = torch.zeros(self.n_nodes, self.n_nodes).bool()
        adj[self.data.edge_index[0], self.data.edge_index[1]] = True

        emb = np.memmap(f'output/{self.dataset}/bert.emb{self.stage}',
                        mode='r',
                        dtype=np.float32,
                        shape=(self.data.x.shape[0], 768))
        features = torch.Tensor(np.array(emb))
        sim = torch.matmul(features, features.T)

        edge_sim = sim[src, dst]
        num_edges = self.data.edge_index.size(1)
        n_remove = int(0.01*num_edges)
        sampled = torch.randperm(n_remove)[:n_remove]
        print("num_edges: ", self.data.edge_index.size(1))
        print(f"delete: {n_remove} edges")
        _, indices = torch.topk(-edge_sim, k=n_remove)
        
        indices = indices[sampled]
        src, dst = self.data.edge_index
        adj[src[indices], dst[indices]] = False
        self.data.edge_index = adj.nonzero().T
        print(self.data)


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
