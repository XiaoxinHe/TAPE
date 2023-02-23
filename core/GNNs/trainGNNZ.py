import argparse
from core.GNNs.gnnz_trainer import GNNZTrainer


if __name__ == "__main__":
    # ! Load data and train
    parser = argparse.ArgumentParser(description='infLM')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--stage', type=int, default=0)
    args = parser.parse_args()

    trainer = GNNZTrainer(args.device, args.stage)
    trainer.train()
    trainer.eval_and_save()
