import argparse
from core.GNNs.gnn_trainer import GNNTrainer


if __name__ == "__main__":
    # ! Load data and train
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_layers', type=int, default=4)
    args = parser.parse_args()

    print(f"\n\n[GNN/{args.stage}]")
    trainer = GNNTrainer(args)
    trainer.train()
    trainer.eval_and_save()
