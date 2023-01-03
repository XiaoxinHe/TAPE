import time
import numpy as np
import random
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from core.preprocess import preprocessing
from core.log import config_logger
from core.model import BertClassifier
from core.gnn import GNN

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
    writer, logger = config_logger(cfg)
    data, token_id, attention_masks = preprocessing(cfg.dataset)
    dataset = TensorDataset(token_id, attention_masks)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        sampler=SequentialSampler(dataset),
        batch_size=BATCH_SIZE
    )

    NOUT = data.y.unique().size(0)
    HIDDEN_SIZE = 128
    seeds = [41, 95, 12, 35]
    train_losses = []
    train_perfs = []
    vali_perfs = []
    test_perfs = []
    per_epoch_times = []
    total_times = []

    for run in range(cfg.train.runs):
        set_seed(seeds[run])

        lm = BertClassifier(feat_shrink=HIDDEN_SIZE).to(cfg.device)
        optimizer_lm = torch.optim.Adam(lm.parameters(), lr=cfg.train.lr_lm)

        start_outer = time.time()
        per_epoch_time = []
        best_val_stage = best_test_stage = float('-inf')

        for stage in range(1, cfg.train.stages+1):
            start = time.time()
            data = data.to(cfg.device)

            data.x = lm.generate_node_features(
                dataloader, cfg.device).to(cfg.device)
            gnn = GNN(nhid=HIDDEN_SIZE, nout=NOUT, gnn_type=cfg.model.gnn_type,
                      nlayer=cfg.model.gnn_nlayer, dropout=cfg.train.dropout, res=cfg.model.res).to(cfg.device)
            optimizer = torch.optim.Adam(gnn.parameters(), lr=cfg.train.lr_gnn)

            best_val_epoch = best_test_epoch = float('-inf')
            for epoch in range(1, cfg.train.epochs+1):
                loss = train_gnn(gnn, data, optimizer)
                train_acc, val_acc, test_acc = test_gnn(gnn, data)
                if val_acc > best_val_epoch:
                    best_val_epoch = val_acc
                    best_test_epoch = test_acc
                    x = gnn(data.x, data.edge_index, readout=False).detach()
                print(
                    f'Epoch: {epoch:02d} / Stage: {stage:02d}, Loss: {loss:.4f}, '
                    f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
                    f'Best Val Acc: {best_val_epoch:.4f}, Best Test Acc: {best_test_epoch:.4f}')
            if best_val_epoch > best_val_stage:
                best_val_stage = best_val_epoch
                best_test_stage = best_test_epoch

            lm_loss = train_lm(lm, dataloader, x, optimizer_lm, cfg.device)
            time_cur_epoch = time.time() - start
            per_epoch_time.append(time_cur_epoch)

            print(f'Stage: {stage:02d}, Loss(LM): {lm_loss:.4f}, '
                  f'Best Val Acc: {best_val_stage:.4f}, Best Test Acc: {best_test_stage:.4f}')

            writer.add_scalar(f'Run{run}/train-lm-loss', lm_loss, stage)
            writer.add_scalar(f'Run{run}/train-acc', train_acc, stage)
            writer.add_scalar(f'Run{run}/val-acc', best_val_stage, stage)
            writer.add_scalar(f'Run{run}/test-acc', best_test_stage, stage)

        per_epoch_time = np.mean(per_epoch_time)
        total_time = (time.time()-start_outer)/3600
        print("Run: ", run)
        print("Train Loss: {:.4f}".format(lm_loss))
        print("Train Accuracy: {:.4f}".format(train_acc))
        print("Vali Accuracy: {:.4f}".format(best_val_stage))
        print("Test Accuracy: {:.4f}".format(best_test_stage))
        print("AVG TIME PER EPOCH: {:.4f} s".format(per_epoch_time))
        print("TOTAL TIME TAKEN: {:.4f} h\n".format(total_time))

        train_losses.append(lm_loss)
        train_perfs.append(train_acc)
        vali_perfs.append(best_val_stage)
        test_perfs.append(best_test_stage)
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


def run_baseline(cfg, train, test):
    writer, logger = config_logger(cfg)
    data = preprocessing(cfg.dataset, use_text=False)

    train_losses = []
    train_perfs = []
    vali_perfs = []
    test_perfs = []
    per_epoch_times = []
    total_times = []

    nout = data.y.unique().size(0)
    seeds = [41, 95, 12, 35]
    for run in range(cfg.train.runs):
        set_seed(seeds[run])

        start_outer = time.time()
        per_epoch_time = []
        best_val_acc = best_test_acc = float('-inf')
        gnn = GNN(nhid=data.x.shape[1], nout=nout, gnn_type=cfg.model.gnn_type,
                  nlayer=cfg.model.gnn_nlayer, dropout=cfg.train.dropout, res=cfg.model.res).to(cfg.device)
        optimizer = torch.optim.Adam(gnn.parameters(), lr=cfg.train.lr_gnn)

        data = data.to(cfg.device)
        best_val_acc = best_test_acc = float('-inf')
        for epoch in range(1, cfg.train.epochs+1):
            start = time.time()
            loss = train(gnn, data, optimizer)
            train_acc, val_acc, test_acc = test(gnn, data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
            time_cur_epoch = time.time() - start
            per_epoch_time.append(time_cur_epoch)

            print(f'Epoch: {epoch:02d}, Train Loss: {loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {best_test_acc:.4f}, '
                  f'Time: {time.time()-start:.4f}')

            writer.add_scalar(f'Run{run}/train-acc', train_acc, epoch)
            writer.add_scalar(f'Run{run}/val-acc', val_acc, epoch)
            writer.add_scalar(f'Run{run}/test-acc', best_test_acc, epoch)

        per_epoch_time = np.mean(per_epoch_time)
        total_time = (time.time()-start_outer)/3600
        print("Run: ", run)
        print("Train Loss: {:.4f}".format(loss))
        print("Train Accuracy: {:.4f}".format(train_acc))
        print("Vali Accuracy: {:.4f}".format(val_acc))
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
