import argparse
from core.GNNs.gamma_trainer import GammaTrainer


if __name__ == "__main__":
    # ! Load data and train
    parser = argparse.ArgumentParser(description='infLM')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--penalty', type=float, default=0.5)
    args = parser.parse_args()

    print(f"\n\n[Gamma/{args.stage}]")
    print(args)
    trainer = GammaTrainer(args)
    trainer.update()
