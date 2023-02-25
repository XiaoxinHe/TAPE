import numpy as np
import torch

from core.preprocess import preprocessing
from core.utils.function.os_utils import init_path
from core.utils.function.np_utils import save_memmap

feat_shrink = 128


class GammaTrainer():
    def __init__(self, args):
        self.device = args.device
        self.epochs = 200
        self.stage = args.stage
        self.dataset = args.dataset
        self.penalty = 0.5

        # ! Load data

    def update(self):

        data = preprocessing(self.dataset, use_text=False)
        lm_x = np.memmap(f"output/{self.dataset}/bert.emb{self.stage}", mode='r',
                         dtype=np.float32, shape=(data.x.shape[0], feat_shrink))
        lm_x = torch.Tensor(np.array(lm_x))

        if self.stage > 0:
            z = np.memmap(f"output/{self.dataset}/z.emb{self.stage}", mode='r',
                          dtype=np.float32, shape=(data.x.shape[0], feat_shrink))
            z = torch.Tensor(np.array(z))

            gamma = np.memmap(f"output/{self.dataset}/gamma.emb{self.stage-1}", mode='r',
                              dtype=np.float32, shape=(data.x.shape[0], feat_shrink))
            gamma = torch.Tensor(np.array(gamma))
            gamma = gamma + self.penalty*(z-lm_x)
        else:
            gamma = torch.zeros_like(lm_x)
        
        print(gamma)
        save_memmap(gamma.cpu().numpy(), init_path(
            f"output/{self.dataset}/gamma.emb{self.stage}"), dtype=np.float32)
