from core.GNNs.gnn_trainer import GNNTrainer


if __name__ == "__main__":
    # ! Load data and train
    device = 0
    trainer = GNNTrainer(device)
    trainer.train()
    trainer.eval_and_save()
