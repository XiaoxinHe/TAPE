from core.GNNs.z_trainer import ZTrainer
import argparse


if __name__ == "__main__":
    # ! Load data and train
    parser = argparse.ArgumentParser(description='infLM')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cora')
    args = parser.parse_args()

    print(f"\n\n[Z/{args.stage}]")
    trainer = ZTrainer(args)
    if args.stage > 0:
        trainer.train()
        trainer.eval_and_save()
    else:
        trainer.save()
