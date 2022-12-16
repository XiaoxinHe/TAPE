import time
import numpy as np
import random
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from core.preprocess import preprocessing, generate_node_embedding
from core.model import BertClassifier
from core.log import config_logger

BATCH_SIZE = 32


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(cfg, train_gnn, train_lm):
    writer, logger, config_string = config_logger(cfg)
    set_seed(cfg.seed)
    data, token_id, attention_masks = preprocessing(cfg.dataset)
    dataset = TensorDataset(token_id, attention_masks)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        sampler=SequentialSampler(dataset),
        batch_size=BATCH_SIZE
    )

    lm = BertClassifier(128)
    features = generate_node_embedding(lm, dataloader, cfg.device)
    optimizer_lm = torch.optim.Adam(lm.parameters(), lr=cfg.train.lr_lm)
    best_test_perf = float('-inf')
    for epoch in range(1, cfg.train.epochs_lm+1):
        start = time.time()
        data.x = features
        features, test_perf = train_gnn(cfg, data)
        if test_perf > best_test_perf:
            best_test_perf = test_perf
        lm = lm.to(cfg.device)
        features = train_lm(lm, dataloader, features, optimizer_lm, cfg.device)
        print(
            f'[!] Epoch: {epoch:02d}, Best Test Acc So Far: {best_test_perf:.4f}, Time: {time.time()-start}')
