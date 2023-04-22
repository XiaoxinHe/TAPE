import argparse
from core.LMs.lm_trainer import LMTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LM training')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="ogbn-arxiv")
    parser.add_argument('--model', type=str, default="microsoft/deberta-base")
    parser.add_argument('--feat_shrink', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=9)
    parser.add_argument('--grad_acc_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--warmup_epochs', type=float, default=0.6)
    parser.add_argument('--eval_patience', type=int, default=50000)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--att_dropout', type=float, default=0.1)
    parser.add_argument('--cla_dropout', type=float, default=0.4)
    parser.add_argument('--use_gpt', action='store_true')

    args = parser.parse_args()
    print(args)

    trainer = LMTrainer(args)
    trainer.train()
    trainer.eval_and_save()
