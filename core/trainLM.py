from core.config import cfg, update_cfg
from core.LMs.lm_trainer import LMTrainer


def run(cfg):
    trainer = LMTrainer(cfg)
    trainer.train()
    trainer.eval_and_save()


if __name__ == '__main__':
    cfg.merge_from_file('configs/cora.yaml')
    cfg = update_cfg(cfg)
    run(cfg)
