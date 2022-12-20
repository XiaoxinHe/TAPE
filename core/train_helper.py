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


def run(cfg, train_gnn, test_gnn, train_lm):
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

    lm = BertClassifier(feat_shrink=128)
    x = generate_node_embedding(lm, dataloader, cfg.device)
    data.x = x
    optimizer_lm = torch.optim.Adam(lm.parameters(), lr=cfg.train.lr_lm)
    best_val_acc = best_test_acc = float('-inf')
    for epoch in range(1, cfg.train.stages+1):
        start = time.time()
        gnn = train_gnn(cfg, data)
        train_acc, val_acc, test_acc = test_gnn(gnn, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        x = gnn(data.x, data.edge_index, False).detach()
        data.x = x
        lm = lm.to(cfg.device)
        x = train_lm(lm, dataloader, data, optimizer_lm, cfg.device)
        data.x = x
        print(
            f'[!] Stage: {epoch:02d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {best_test_acc:.4f}, Time: {time.time()-start:.4f}')
