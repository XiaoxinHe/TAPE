import pandas as pd
import argparse
from core.GNNs.gnn_trainer import GNNTrainer
from core.GNNs.dgl_gnn_trainer import DGLGNNTrainer
from core.GNNs.ensemble_trainer import EnsembleTrainer
import time

if __name__ == "__main__":
    # ! Load data and train
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='ogbn-arxiv')
    parser.add_argument('--combine', type=str, default='concat')
    parser.add_argument('--gnn_model_name', type=str, default='GCN')
    parser.add_argument('--lm_model_name', type=str,
                        default='microsoft/deberta-base')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--input_norm', type=str, default='T')
    parser.add_argument('--runs', type=int, default=4)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--use_dgl', action='store_true')
    parser.add_argument('--use_ogb', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0.00)

    args = parser.parse_args()
    print(args)

    t0 = time.time()
    ensembler = EnsembleTrainer(args)
    f1_acc = []
    f2_acc = []
    # all_acc = {'f1': [], 'f2': [], 'ensemble': []}
    all_acc = {'f1': [], 'f2': [], 'f3': [], 'ensemble': []}
    TRAINER = DGLGNNTrainer if args.use_dgl else GNNTrainer
    for seed in range(args.runs):
        all_pred = []
        accs = {}
        for combine in ['f1', 'f2', 'f3']:
            # for combine in ['f1', 'f2']:
            args.combine = combine
            args.seed = seed
            trainer = TRAINER(args)
            trainer.train()
            pred, acc = trainer.eval_and_save()
            all_acc[combine].append(acc)
            all_pred.append(pred)
        pred_ensemble = sum(all_pred)/len(all_pred)
        all_acc['ensemble'].append(ensembler.eval(pred_ensemble))

    for k, v in all_acc.items():
        df = pd.DataFrame(v)
        print(
            f"{k} val acc: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f} test acc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")

    print(f"Total training time: {time.time()-t0:.2f}s")
