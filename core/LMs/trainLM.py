from core.LMs.lm_trainer import LMTrainer

CKPT = "output/stage0.pt"

if __name__ == "__main__":
    # ! Init Arguments
    trainer = LMTrainer(CKPT)

    # ! Load data and train
    trainer.train()
