import numpy as np
import torch
from torch_geometric.transforms import ToSparseTensor, ToUndirected
from core.model_utils.EnGCN import EnGCN
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from core.preprocess import preprocessing


def load_data(dataset_name):

    data = preprocessing(dataset_name)
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
    return data, x, y, split_masks, evaluator, processed_dir


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
        ) = load_data(args.dataset)

        print('load from OGB feature!')
        if self.type_model == "EnGCN":
            self.model = EnGCN(
                args,
                self.data,
                self.evaluator,
            )
        else:
            raise NotImplementedError
        self.model.to(self.device)

        if len(list(self.model.parameters())) != 0:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        else:
            self.optimizer = None

    def train_ensembling(self, seed):
        # assert isinstance(self.model, (SAdaGCN, AdaGCN, GBGCN))
        input_dict = self.get_input_dict(0)
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
