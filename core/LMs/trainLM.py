import argparse
from core.LMs.lm_trainer import LMTrainer


if __name__ == "__main__":
    # ! Load data and train
    parser = argparse.ArgumentParser(description='infLM')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="cora")
    args = parser.parse_args()

    trainer = LMTrainer(args)
    print(f"\n\n[LM/{args.stage}]")
    # ! Load data and train
    trainer.train()
    trainer.eval_and_save()
