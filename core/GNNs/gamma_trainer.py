from core.GNNs.kd_gnn_trainer import load_data
import numpy as np
import torch

from core.utils.function.os_utils import init_path
from core.utils.function.np_utils import save_memmap

feat_shrink = ""


class GammaTrainer():
    def __init__(self, args):
        self.device = args.device
        self.epochs = 200
        self.stage = args.stage
        self.dataset = args.dataset
        self.penalty = args.penalty
        self.dim = feat_shrink if feat_shrink else 768
        self.ckpt = init_path(f"output/{self.dataset}/gamma.emb{self.stage}")

    def update(self):

        data = load_data(self.dataset)

        if self.stage > 0:
            lm_x = np.memmap(f"output/{self.dataset}/bert.emb{self.stage}",
                             mode='r',
                             dtype=np.float32,
                             shape=(data.x.shape[0], self.dim))
            lm_x = torch.Tensor(np.array(lm_x))
            z = np.memmap(f"output/{self.dataset}/z.emb{self.stage}",
                          mode='r',
                          dtype=np.float32,
                          shape=(data.x.shape[0], self.dim))
            z = torch.Tensor(np.array(z))

            gamma = np.memmap(f"output/{self.dataset}/gamma.emb{self.stage-1}",
                              mode='r',
                              dtype=np.float32,
                              shape=(data.x.shape[0], self.dim))
            gamma = torch.Tensor(np.array(gamma))
            gamma = gamma + self.penalty*(z-lm_x)
        else:
            gamma = torch.zeros(data.x.shape[0], self.dim)

        print(gamma)
        save_memmap(gamma.cpu().numpy(), self.ckpt, dtype=np.float32)
