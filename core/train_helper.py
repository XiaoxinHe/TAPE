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
    data, token_id, attention_masks = preprocessing(cfg.dataset)
    dataset = TensorDataset(token_id, attention_masks)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        sampler=SequentialSampler(dataset),
        batch_size=BATCH_SIZE
    )

    train_losses = []
    train_perfs = []
    vali_perfs = []
    test_perfs = []
    per_epoch_times = []
    total_times = []

    seeds = [41, 95, 12, 35]
    for run in range(cfg.train.runs):
        set_seed(seeds[run])

        lm = BertClassifier(feat_shrink=128)
        optimizer_lm = torch.optim.Adam(lm.parameters(), lr=cfg.train.lr_lm)

        start_outer = time.time()
        per_epoch_time = []
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
            time_cur_epoch = time.time() - start
            per_epoch_time.append(time_cur_epoch)

            print(f'[!] Stage: {epoch:02d}, Train Loss(LM): {loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {best_val_acc:.4f}, Test Acc: {best_test_acc:.4f}, '
                  f'Time: {time.time()-start:.4f}\n')

            writer.add_scalar(f'Run{run}/train-lm-loss', loss, epoch)
            writer.add_scalar(f'Run{run}/train-acc', train_acc, epoch)
            writer.add_scalar(f'Run{run}/val-acc', best_val_acc, epoch)
            writer.add_scalar(f'Run{run}/test-acc', best_test_acc, epoch)

        per_epoch_time = np.mean(per_epoch_time)
        total_time = (time.time()-start_outer)/3600
        print("Run: ", run)
        print("Train Loss: {:.4f}".format(loss))
        print("Train Accuracy: {:.4f}".format(train_acc))
        print("Vali Accuracy: {:.4f}".format(best_val_acc))
        print("Test Accuracy: {:.4f}".format(best_test_acc))
        print("AVG TIME PER EPOCH: {:.4f} s".format(per_epoch_time))
        print("TOTAL TIME TAKEN: {:.4f} h\n".format(total_time))

        train_losses.append(loss)
        train_perfs.append(train_acc)
        vali_perfs.append(best_val_acc)
        test_perfs.append(best_test_acc)
        per_epoch_times.append(per_epoch_time)
        total_times.append(total_time)

    if cfg.train.runs > 1:
        train_loss = torch.tensor(train_losses)
        train_perf = torch.tensor(train_perfs)
        vali_perf = torch.tensor(vali_perfs)
        test_perf = torch.tensor(test_perfs)
        per_epoch_time = torch.tensor(per_epoch_times)
        total_time = torch.tensor(total_times)
        print(f'\nFinal Train Loss: {train_loss.mean():.4f} ± {train_loss.std():.4f}'
              f'\nFinal Train: {train_perf.mean():.4f} ± {train_perf.std():.4f}'
              f'\nFinal Vali: {vali_perf.mean():.4f} ± {vali_perf.std():.4f}'
              f'\nFinal Test: {test_perf.mean():.4f} ± {test_perf.std():.4f}'
              f'\nSeconds/epoch: {per_epoch_time.mean():.4f}'
              f'\nHours/total: {total_time.mean():.4f}')
        logger.info("-"*50)
        logger.info(cfg)
        logger.info(f'\nFinal Train Loss: {train_loss.mean():.4f} ± {train_loss.std():.4f}'
                    f'\nFinal Train: {train_perf.mean():.4f} ± {train_perf.std():.4f}'
                    f'\nFinal Vali: {vali_perf.mean():.4f} ± {vali_perf.std():.4f}'
                    f'\nFinal Test: {test_perf.mean():.4f} ± {test_perf.std():.4f}'
                    f'\nSeconds/epoch: {per_epoch_time.mean():.4f}'
                    f'\nHours/total: {total_time.mean():.4f}')


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
