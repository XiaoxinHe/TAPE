import argparse
from core.LMs.kd_lm_trainer import KDLMTrainer as LMTrainer


if __name__ == "__main__":
    # ! Load data and train
    parser = argparse.ArgumentParser(description='infLM')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="cora")
    parser.add_argument('--pl_weight', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=3e-5)

    args = parser.parse_args()
    print(f"\n\n[LM/{args.stage}]")
    print(args)
    trainer = LMTrainer(args)

    # ! Load data and train
    trainer.train()
    trainer.eval_and_save()
