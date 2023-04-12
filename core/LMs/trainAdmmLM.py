import argparse
from core.LMs.admm_lm_trainer import AdmmLMTrainer


if __name__ == "__main__":
    # ! Load data and train
    parser = argparse.ArgumentParser(description='infLM')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="cora")
    parser.add_argument('--penalty', type=float, default="0.5")
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup_epochs', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--att_dropout', type=float, default=0.1)
    parser.add_argument('--cla_dropout', type=float, default=0.1)

    args = parser.parse_args()
    print(f"\n\n[LM/{args.stage}]")
    print(args)
    trainer = AdmmLMTrainer(args)

    # ! Load data and train
    trainer.train()
    trainer.eval_and_save()
