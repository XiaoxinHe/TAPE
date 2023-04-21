import argparse
from core.GNNs.gnn_trainer import GNNTrainer
from core.GNNs.dgl_gnn_trainer import DGLGNNTrainer
import pandas as pd

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
    parser.add_argument('--use_dgl', action='store_true')
    parser.add_argument('--use_ogb', action='store_true')

    args = parser.parse_args()
    print(args)

    res = []
    for _ in range(args.runs):
        if args.use_dgl:
            trainer = DGLGNNTrainer(args)
        else:
            trainer = GNNTrainer(args)
        trainer.train()
        res.append(trainer.eval_and_save())
    df = pd.DataFrame(res)

    print(f"val acc: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}")
    print(
        f"test acc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")
