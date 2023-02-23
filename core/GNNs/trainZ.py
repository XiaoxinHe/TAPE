import argparse
from core.GNNs.z_trainer import ZTrainer


if __name__ == "__main__":
    # ! Load data and train
    parser = argparse.ArgumentParser(description='infLM')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--stage', type=int, default=0)
    args = parser.parse_args()
    
    
    trainer = ZTrainer(args.device, args.stage)
    trainer.train()
    trainer.eval_and_save()
    
    
