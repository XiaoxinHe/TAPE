import argparse
from core.GNNs.z_trainer import ZTrainer


if __name__ == "__main__":
    # ! Load data and train
    parser = argparse.ArgumentParser(description='infLM')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cora')
    args = parser.parse_args()

    trainer = ZTrainer(args)
    print(f"\n\n[Z/{args.stage}]")
    if args.stage > 0:
        trainer.train()
        trainer.eval_and_save()
    else:
        trainer.save()
