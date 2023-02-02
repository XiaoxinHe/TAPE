
import numpy as np
import torch
from torch_geometric.transforms import ToSparseTensor
from core.model_utils.EnGCN import EnGCN
from ogb.nodeproppred import Evaluator
from core.model import BertClassifierV2
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from train.v3 import pretrain_lm, test_lm
import time

BATCH_SIZE = 32


def load_data(dataset_name):
    from core.preprocess import preprocessing
    data, token_id, attention_masks = preprocessing(dataset_name)
    if "ogbn" not in dataset_name:
        trans = ToSparseTensor()
        data = trans(data)

    x = data.x
    split_masks = {}
    split_masks['train'] = data.train_mask
    split_masks['valid'] = data.val_mask
    split_masks['test'] = data.test_mask
    if "ogb" in dataset_name:
        evaluator = Evaluator(name=dataset_name)
        y = data.y = data.y.squeeze()
    else:
        evaluator = None
        y = data.y

    processed_dir = 'dataset/'+dataset_name+'/processed'

    return data, x, y, split_masks, evaluator, processed_dir, token_id, attention_masks


class trainer(object):
    def __init__(self, args):
        self.dataset = args.dataset

        self.device = torch.device(
            f"cuda:{args.cuda_num}" if args.cuda else "cpu")
        self.args = args
        self.args.device = self.device

        self.type_model = args.type_model
        self.epochs = args.epochs
        self.eval_steps = args.eval_steps

        # used to indicate multi-label classification.
        # If it is, using BCE and micro-f1 performance metric
        self.multi_label = args.multi_label
        if self.multi_label:
            self.loss_op = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_op = torch.nn.NLLLoss()

        (
            self.data,
            self.x,
            self.y,
            self.split_masks,
            self.evaluator,
            self.processed_dir,
            token_id,
            attention_masks
        ) = load_data(args.dataset)

        self.lm = BertClassifierV2(
            feat_shrink=args.num_feats, nout=args.num_classes).to(self.device)

        self.optimizer_lm = torch.optim.Adam(self.lm.parameters(), lr=1e-5)

        dataset = TensorDataset(token_id, attention_masks)
        self.dataloader = DataLoader(
            dataset,
            shuffle=False,
            sampler=SequentialSampler(dataset),
            batch_size=BATCH_SIZE
        )
        print("[!] Pretraining LM")
        start = time.time()
        self.data = self.data.to(self.device)
        loss = pretrain_lm(self.lm, self.dataloader,
                           self.data, self.optimizer_lm, self.device)
        train_acc, val_acc, test_acc = test_lm(
            self.lm, self.dataloader, self.data, self.split_masks, self.evaluator, self.device)
        print(f'Loss: {loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {time.time()-start:.4f}\n')

        self.model = EnGCN(
            args,
            self.data,
            self.evaluator
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    def train_ensembling(self, seed):
        # assert isinstance(self.model, (SAdaGCN, AdaGCN, GBGCN))
        input_dict = self.get_input_dict(0)
        input_dict["x"] = self.lm.generate_node_features(
            self.dataloader, self.device)
        acc = self.model.train_and_test(input_dict)
        return acc

    def get_input_dict(self, epoch):
        if self.type_model in [
            "EnGCN",
        ]:
            input_dict = {
                "split_masks": self.split_masks,
                "data": self.data,
                "x": self.x,
                "y": self.y,
                "optimizer": self.optimizer,
                "loss_op": self.loss_op,
                "device": self.device,
            }
        else:
            Exception(
                f"the model of {self.type_model} has not been implemented")
        return input_dict
