from core.config import cfg, update_cfg
from core.GNNs.ensemble_trainer import EnsembleTrainer
import pandas as pd


def run(cfg):
    all_acc = []
    for seed in range(cfg.gnn.train.runs):
        cfg.seed = seed
        ensembler = EnsembleTrainer(cfg)
        acc = ensembler.train()
        all_acc.append(acc)

    df = pd.DataFrame(all_acc)

    for f in df.keys():
        df_ = pd.DataFrame([r for r in df[f]])
        print(
            f"[{f}] ValACC: {df_['val_acc'].mean():.4f} ± {df_['val_acc'].std():.4f}, TestAcc: {df_['test_acc'].mean():.4f} ± {df_['test_acc'].std():.4f}")


if __name__ == '__main__':
    cfg.merge_from_file('configs/cora.yaml')
    cfg = update_cfg(cfg)
    run(cfg)
