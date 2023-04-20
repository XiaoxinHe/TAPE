import argparse
from core.GNNs.gnn_trainer import GNNTrainer


if __name__ == "__main__":
    # ! Load data and train
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='pubmed')
    parser.add_argument('--combine', type=str, default='f1')
    parser.add_argument('--gnn_model_name', type=str, default='GCN')
    parser.add_argument('--lm_model_name', type=str,
                        default='microsoft/deberta-base')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--input_norm', type=str, default='T')

    args = parser.parse_args()

    trainer = GNNTrainer(args)
    trainer.train()
    trainer.eval_and_save()
