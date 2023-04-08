from core.GNNs.admm_gnn_trainer import ADMMGNNTrainer as GNNTrainer
import argparse


if __name__ == "__main__":
    # ! Load data and train
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--beta', type=float, default=1e-3)

    args = parser.parse_args()

    print(f"\n\n[GNN/{args.stage}]")
    print(args)
    trainer = GNNTrainer(args)
    trainer.train()
    trainer.eval_and_save()
