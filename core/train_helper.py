import time
import numpy as np
import random
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from core.preprocess import preprocessing
from core.log import config_logger
from core.model import BertClassifier, Z
from core.gnn import GCN
#from core.sage import SAGE
from ogb.nodeproppred import Evaluator

BATCH_SIZE = 32


def load_data(dataset_name, use_text):
    if use_text:
        data, token_id, attention_masks = preprocessing(dataset_name, use_text)
    else:
        data = preprocessing(dataset_name, use_text)
        token_id = None
        attention_masks = None

    split_masks = {}
    split_masks['train'] = data.train_mask
    split_masks['valid'] = data.val_mask
    split_masks['test'] = data.test_mask

    if "ogbn" in dataset_name:
        x = data.x
        y = data.y = data.y.squeeze()
        evaluator = Evaluator(name=dataset_name)
    else:
        x = data.x
        y = data.y
        evaluator = None
    processed_dir = 'dataset/'+dataset_name+'/processed'

    return data, x, y, split_masks, evaluator, processed_dir, token_id, attention_masks


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_baseline(cfg, train, test, train_lm, pretrain_lm=None, test_lm=None):
    writer, logger = config_logger(cfg)

    data, x, y, split_masks, evaluator, processed_dir, token_id, attention_masks = load_data(
        cfg.dataset, use_text=cfg.use_text)
    NOUT = data.y.unique().size(0)

    if cfg.use_text:
        dataset = TensorDataset(token_id, attention_masks)
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            sampler=SequentialSampler(dataset),
            batch_size=BATCH_SIZE
        )
        lm = BertClassifier(feat_shrink=cfg.model.nhid,
                            nout=NOUT).to(cfg.device)
        if pretrain_lm is not None:
            print("[!] Pretraining LM")
            LM_PATH = f"/home/xiaoxin/TAG/.cache/baseline_LM_{cfg.dataset}_{cfg.logfile}.pt"
            optimizer_lm = torch.optim.Adam(
                lm.parameters(), lr=cfg.train.lr_lm)
            best_val = 0
            start = time.time()
            for epoch in range(cfg.train.epochs_ft):
                loss = pretrain_lm(lm, dataloader, data,
                                   optimizer_lm, cfg.device)
                train_acc, val_acc, test_acc = test_lm(
                    lm, dataloader, data, split_masks, evaluator, cfg.device)
                print(f'[LM] Epoch: {epoch}, Loss: {loss:.4f}, '
                      f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {time.time()-start:.4f}')
                if val_acc > best_val:
                    best_val = val_acc
                    if not os.path.exists(".cache/"):
                        os.mkdir(".cache/")
                    torch.save(lm.state_dict(), LM_PATH)
            lm.load_state_dict(torch.load(LM_PATH))

        x = lm.generate_node_features(dataloader, cfg.device).to(cfg.device)
        data.x = x

    train_losses = []
    train_perfs = []
    vali_perfs = []
    test_perfs = []
    per_epoch_times = []
    total_times = []

    seeds = [41, 95, 12, 35]
    for run in range(cfg.train.runs):
        set_seed(seeds[run])

        start_outer = time.time()
        per_epoch_time = []
        best_val_acc = best_test_acc = float('-inf')
        gnn = GCN(in_channels=data.x.shape[1], hidden_channels=cfg.model.nhid, out_channels=NOUT,
                  num_layers=cfg.model.gnn_nlayer, dropout=cfg.train.dropout).to(cfg.device)
        optimizer = torch.optim.Adam(gnn.parameters(), lr=cfg.train.lr_gnn)

        data = data.to(cfg.device)
        best_val_acc = best_test_acc = float('-inf')
        for epoch in range(1, cfg.train.epochs+1):
            start = time.time()
            loss = train(gnn, data, optimizer)
            train_acc, val_acc, test_acc = test(
                gnn, data, split_masks, evaluator)
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


def run(cfg, train_gnn, test_gnn, train_lm, pretrain_lm=None, test_lm=None):
    writer, logger = config_logger(cfg)

    data, x, y, split_masks, evaluator, processed_dir, token_id, attention_masks = load_data(
        cfg.dataset, use_text=True)

    dataset = TensorDataset(token_id, attention_masks)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        sampler=SequentialSampler(dataset),
        batch_size=BATCH_SIZE
    )

    split_masks = {}
    split_masks['train'] = data.train_mask
    split_masks['valid'] = data.val_mask
    split_masks['test'] = data.test_mask

    NOUT = data.y.unique().size(0)
    seeds = [41, 95, 12, 35]
    train_losses = []
    train_perfs = []
    vali_perfs = []
    test_perfs = []
    per_epoch_times = []
    total_times = []
    LM_PATH = f"/home/xiaoxin/TAG/.cache/v4_LM_{cfg.dataset}_{cfg.logfile}.pt"
    for run in range(cfg.train.runs):
        set_seed(seeds[run])

        lm = BertClassifier(feat_shrink=cfg.model.nhid,
                            nout=NOUT).to(cfg.device)
        optimizer_lm = torch.optim.Adam(lm.parameters(), lr=cfg.train.lr_lm)
        data = data.to(cfg.device)

        if pretrain_lm is not None:
            best_val = 0
            print("[!] Pretraining LM")
            start = time.time()
            for epoch in range(cfg.train.epochs_ft):
                loss = pretrain_lm(lm, dataloader, data,
                                   optimizer_lm, cfg.device)
                train_acc, val_acc, test_acc = test_lm(
                    lm, dataloader, data, split_masks, evaluator, cfg.device)
                print(f'[LM] Epoch: {epoch}, Loss: {loss:.4f}, '
                      f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {time.time()-start:.4f}')
                if val_acc > best_val:
                    best_val = val_acc
                    if not os.path.exists(".cache/"):
                        os.mkdir(".cache/")
                    torch.save(lm.state_dict(), LM_PATH)
        lm.load_state_dict(torch.load(LM_PATH))
        lm_z = lm.generate_node_features(dataloader, cfg.device).to(cfg.device)
        model_z = Z(z=lm_z.detach().clone()).to(cfg.device)

        gnn = GCN(in_channels=cfg.model.nhid, hidden_channels=cfg.model.nhid, out_channels=NOUT,
                  num_layers=cfg.model.gnn_nlayer, dropout=cfg.train.dropout).to(cfg.device)
        start_outer = time.time()
        per_epoch_time = []
        # best_val_stage = best_test_stage = float('-inf')
        best_val = best_test = float('-inf')

        for stage in range(1, cfg.train.stages+1):

            optimizer_gnn = torch.optim.Adam(
                gnn.parameters(), lr=cfg.train.lr_gnn)
            optimizer_z = torch.optim.Adam(
                model_z.parameters(), lr=cfg.train.lr_z)
            optimizer_lm = torch.optim.Adam(
                lm.parameters(), lr=cfg.train.lr_lm)

            start_stage = time.time()
            data = data.to(cfg.device)
            # best_val_epoch = best_test_epoch = float('-inf')
            if stage > 1:
                lm_z = lm.generate_node_features(
                    dataloader, cfg.device).to(cfg.device)

            for epoch in range(1, cfg.train.epochs+1):
                start = time.time()
                new_best_str = ''
                loss, loss_gnn, loss_z = train_gnn(
                    gnn, model_z, data, lm_z, optimizer_gnn, optimizer_z, cfg.train.alpha)
                train_acc, val_acc, test_acc = test_gnn(
                    gnn, model_z, data, split_masks, evaluator)
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc
                    new_best_str = ' (new best test)'
                    if not os.path.exists(".cache/"):
                        os.mkdir(".cache/")
                    torch.save(
                        gnn.state_dict(),
                        f"./.cache/{cfg.dataset}_{cfg.model.gnn_type}_{cfg.logfile}.pt",
                    )
                    torch.save(
                        model_z.state_dict(),
                        f"./.cache/{cfg.dataset}_model_z_{cfg.logfile}.pt",
                    )
                end = time.time()
                print(
                    f'Stage: {stage:02d}, Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, loss(GNN): {loss_gnn:.4f}, loss(Z): {loss_z:.8f}, '
                    f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
                    f'Time: {(end-start):.4f}, '
                    f'Best Test Acc: {best_test:.4f}{new_best_str}')

            gnn.load_state_dict(torch.load(
                f"./.cache/{cfg.dataset}_{cfg.model.gnn_type}_{cfg.logfile}.pt"))
            model_z.load_state_dict(torch.load(
                f"./.cache/{cfg.dataset}_model_z_{cfg.logfile}.pt"))
            z = model_z().detach()

            start = time.time()
            for epoch in range(1):
                lm_loss, lm_loss0, lm_loss1 = train_lm(
                    lm, dataloader, z, data, optimizer_lm, cfg.train.beta, split_masks, evaluator, cfg.device, LM_PATH)

            lm.load_state_dict(torch.load(LM_PATH))
            train_acc, val_acc, test_acc = test_lm(
                lm, dataloader, data, split_masks, evaluator, cfg.device)
            print(
                f'[LM] Loss: {lm_loss: .4f}, Loss0: {lm_loss0: .4f}, Loss1: {lm_loss1: .4f}, '
                f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {time.time()-start:.4f}')

            time_cur_epoch = time.time() - start_stage
            per_epoch_time.append(time_cur_epoch)

            # writer.add_scalar(f'Run{run}/train-lm-loss', lm_loss, stage)
            # writer.add_scalar(f'Run{run}/train-acc', train_acc, stage)
            # writer.add_scalar(f'Run{run}/val-acc', best_val, stage)
            # writer.add_scalar(f'Run{run}/test-acc', best_test, stage)

        per_epoch_time = np.mean(per_epoch_time)
        total_time = (time.time()-start_outer)/3600
        print("Run: ", run)
        print("Train Loss: {:.4f}".format(lm_loss))
        print("Train Accuracy: {:.4f}".format(train_acc))
        print("Vali Accuracy: {:.4f}".format(best_val))
        print("Test Accuracy: {:.4f}".format(best_test))
        print("AVG TIME PER EPOCH: {:.4f} s".format(per_epoch_time))
        print("TOTAL TIME TAKEN: {:.4f} h\n".format(total_time))

        train_losses.append(lm_loss)
        train_perfs.append(train_acc)
        vali_perfs.append(best_val)
        test_perfs.append(best_test)
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


def run_v3(cfg, train_gnn, test_gnn, train_lm, pretrain_lm=None, test_lm=None):
    writer, logger = config_logger(cfg)

    data, x, y, split_masks, evaluator, processed_dir, token_id, attention_masks = load_data(
        cfg.dataset, use_text=True)

    dataset = TensorDataset(token_id, attention_masks)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        sampler=SequentialSampler(dataset),
        batch_size=BATCH_SIZE
    )

    split_masks = {}
    split_masks['train'] = data.train_mask
    split_masks['valid'] = data.val_mask
    split_masks['test'] = data.test_mask

    NOUT = data.y.unique().size(0)
    seeds = [41, 95, 12, 35]
    train_losses = []
    train_perfs = []
    vali_perfs = []
    test_perfs = []
    per_epoch_times = []
    total_times = []
    LM_PATH = f"/home/xiaoxin/TAG/.cache/v3_LM_{cfg.dataset}_{cfg.logfile}.pt"
    for run in range(cfg.train.runs):
        set_seed(seeds[run])

        lm = BertClassifier(feat_shrink=cfg.model.nhid,
                            dropout=0.5,
                            nout=NOUT).to(cfg.device)
        optimizer_lm = torch.optim.Adam(lm.parameters(), lr=cfg.train.lr_lm)
        data = data.to(cfg.device)

        if pretrain_lm is not None:
            best_val = 0
            print("[!] Pretraining LM")
            start = time.time()
            for epoch in range(cfg.train.epochs_ft):
                loss = pretrain_lm(lm, dataloader, data,
                                   optimizer_lm, cfg.device)
                train_acc, val_acc, test_acc = test_lm(
                    lm, dataloader, data, split_masks, evaluator, cfg.device)

                print(f'[LM] Epoch: {epoch}, Loss: {loss:.4f}, '
                      f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {time.time()-start:.4f}')
                if val_acc > best_val:
                    best_val = val_acc
                    if not os.path.exists(".cache/"):
                        os.mkdir(".cache/")
                    torch.save(lm.state_dict(), LM_PATH)
        lm.load_state_dict(torch.load(LM_PATH))
        lm_z = lm.generate_node_features(dataloader, cfg.device).to(cfg.device)
        model_z = Z(z=lm_z.detach().clone()).to(cfg.device)

        # gnn = GCN(in_channels=cfg.model.nhid, hidden_channels=cfg.model.nhid, out_channels=NOUT,
        #           num_layers=cfg.model.gnn_nlayer, dropout=cfg.train.dropout).to(cfg.device)

        start_outer = time.time()
        per_epoch_time = []
        best_val = best_test = float('-inf')

        for stage in range(1, cfg.train.stages+1):
            gnn = GCN(in_channels=cfg.model.nhid, hidden_channels=cfg.model.nhid, out_channels=NOUT,
                      num_layers=cfg.model.gnn_nlayer, dropout=cfg.train.dropout).to(cfg.device)
            optimizer_gnn = torch.optim.Adam(
                gnn.parameters(), lr=cfg.train.lr_gnn)
            optimizer_z = torch.optim.Adam(
                model_z.parameters(), lr=cfg.train.lr_z)
            optimizer_lm = torch.optim.Adam(
                lm.parameters(), lr=cfg.train.lr_lm)

            start_stage = time.time()
            data = data.to(cfg.device)

            if stage > 1:
                lm_z = lm.generate_node_features(
                    dataloader, cfg.device).to(cfg.device)

            for epoch in range(1, cfg.train.epochs+1):
                start = time.time()
                new_best_str = ''
                loss, loss_gnn, loss_z = train_gnn(
                    gnn, model_z, data, lm_z, optimizer_gnn, optimizer_z, cfg.train.alpha)
                # data.x = lm_z
                train_acc, val_acc, test_acc = test_gnn(
                    gnn, model_z, data, split_masks, evaluator)
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc
                    new_best_str = ' (new best test)'
                    if not os.path.exists(".cache/"):
                        os.mkdir(".cache/")
                    torch.save(
                        gnn.state_dict(),
                        f"./.cache/{cfg.dataset}_{cfg.model.gnn_type}_{cfg.logfile}.pt",
                    )
                    torch.save(
                        model_z.state_dict(),
                        f"./.cache/{cfg.dataset}_model_z_{cfg.logfile}.pt",
                    )
                end = time.time()
                print(
                    f'Stage: {stage:02d}, Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, loss(GNN): {loss_gnn:.4f}, loss(Z): {loss_z:.8f}, '
                    f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, '
                    f'Time: {(end-start):.4f}, '
                    f'Best Test Acc: {best_test:.4f}{new_best_str}')

            gnn.load_state_dict(torch.load(
                f"./.cache/{cfg.dataset}_{cfg.model.gnn_type}_{cfg.logfile}.pt"))
            model_z.load_state_dict(torch.load(
                f"./.cache/{cfg.dataset}_model_z_{cfg.logfile}.pt"))
            z = model_z().detach()

            start = time.time()
            for epoch in range(1):
                lm_loss = train_lm(lm, dataloader, z, data, optimizer_lm,
                                   split_masks, evaluator, cfg.device, LM_PATH)
            lm.load_state_dict(torch.load(LM_PATH))
            train_acc, val_acc, test_acc = test_lm(
                lm, dataloader, data, split_masks, evaluator, cfg.device)
            print(
                f'[LM] Loss: {lm_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {time.time()-start:.4f}')

            time_cur_epoch = time.time() - start_stage
            per_epoch_time.append(time_cur_epoch)

            # writer.add_scalar(f'Run{run}/train-lm-loss', lm_loss, stage)
            # writer.add_scalar(f'Run{run}/train-acc', train_acc, stage)
            # writer.add_scalar(f'Run{run}/val-acc', best_val, stage)
            # writer.add_scalar(f'Run{run}/test-acc', best_test, stage)

        per_epoch_time = np.mean(per_epoch_time)
        total_time = (time.time()-start_outer)/3600
        print("Run: ", run)
        print("Train Loss: {:.4f}".format(lm_loss))
        print("Train Accuracy: {:.4f}".format(train_acc))
        print("Vali Accuracy: {:.4f}".format(best_val))
        print("Test Accuracy: {:.4f}".format(best_test))
        print("AVG TIME PER EPOCH: {:.4f} s".format(per_epoch_time))
        print("TOTAL TIME TAKEN: {:.4f} h\n".format(total_time))

        train_losses.append(lm_loss)
        train_perfs.append(train_acc)
        vali_perfs.append(best_val)
        test_perfs.append(best_test)
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
