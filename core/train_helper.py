import time
import numpy as np
import random
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from core.preprocess import preprocessing, generate_node_embedding
from core.model import GCN, BertClassifier
from core.log import config_logger
# from core.model import GCN, Z
from core.gnn import GNN

BATCH_SIZE = 64


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

    lm = BertClassifier(feat_shrink=128)
    optimizer_lm = torch.optim.Adam(lm.parameters(), lr=cfg.train.lr_lm)

    best_val_acc = best_test_acc = float('-inf')
    for epoch in range(1, cfg.train.stages+1):
        start = time.time()
        data = data.to(cfg.device)
        lm = lm.to(cfg.device)
        data.x = lm.generate_node_features(dataloader, cfg.device)
        data = data.to(cfg.device)
        gnn = GNN(nhid=128, nout=data.y.unique().size(
            0), gnn_type=cfg.model.gnn_type, nlayer=cfg.model.gnn_nlayer, dropout=0.).to(cfg.device)
        optimizer = torch.optim.Adam(gnn.parameters(), lr=cfg.train.lr_gnn)
        train_acc, val_acc, test_acc, x = train_gnn(data, gnn, optimizer)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        data.x = x
        lm = lm.to(cfg.device)
        loss = train_lm(lm, dataloader, data, optimizer_lm, cfg.device)
        print(f'[!] Stage: {epoch:02d}, Train Loss(LM): {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {best_test_acc:.4f}, Time: {time.time()-start:.4f}\n')


def run_v2(cfg, train_gnn, test_gnn, train_lm):
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
    init_z = lm_z = lm.generate_node_embedding(dataloader, cfg.device)

    # model_z = Z(lm_z).to(cfg.device)
    # gnn = GCN(in_channels=768, hidden_channels=128,
    #           out_channels=data.y.unique().size(0), num_layers=4, dropout=0).to(cfg.device)
    # optimizer_z = torch.optim.Adam(model_z.parameters(), lr=1e-4)
    # optimizer_gnn = torch.optim.Adam(gnn.parameters(), lr=cfg.train.lr_gnn)

    optimizer_lm = torch.optim.Adam(lm.parameters(), lr=cfg.train.lr_lm)
    best_val_acc = best_test_acc = float('-inf')
    for epoch in range(1, cfg.train.stages+1):
        start = time.time()
        lm_z = lm_z.to(cfg.device)
        init_z = init_z.to(cfg.device)
        # gnn = train_gnn(cfg, gnn, model_z, optimizer_gnn, optimizer_z,  data, lm_z)
        val_acc, test_acc, z = train_gnn(cfg, data, init_z, lm_z)
        # train_acc, val_acc, test_acc = test_gnn(gnn, model_z, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        # z = model_z()
        lm = lm.to(cfg.device)
        lm = train_lm(lm, dataloader, z, optimizer_lm, cfg.device)
        lm_z = []
        for batch in dataloader:
            batch = tuple(t.to(cfg.device) for t in batch)
            output = lm(batch)
            lm_z.append(output.detach().cpu())
        lm_z = torch.cat(lm_z, dim=0)
        # print(
        #     f'[!] Stage: {epoch:02d}, Train Acc: {train_acc:.4f}, Val Acc: {best_val_acc:.4f}, Test Acc: {best_test_acc:.4f}, Time: {time.time()-start:.4f}')
        print(
            f'[!] Stage: {epoch:02d}, Val Acc: {best_val_acc:.4f}, Test Acc: {best_test_acc:.4f}, Time: {time.time()-start:.4f}')
