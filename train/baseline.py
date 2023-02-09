import torch
from core.config import cfg, update_cfg
from core.train_helper import run_baseline as run
from ogb.nodeproppred import Evaluator
from train.v3 import pretrain_lm, train_lm, test_lm

def evaluate(out, y, split_mask, evaluator):
    acc = {}
    if evaluator:
        y_true = y.unsqueeze(-1)
        y_pred = out.argmax(dim=-1, keepdim=True)
        for phase in ["train", "valid", "test"]:
            acc[phase] = evaluator.eval(
                {
                    "y_true": y_true[split_mask[phase]],
                    "y_pred": y_pred[split_mask[phase]],
                }
            )["acc"]
    else:
        pred = out.argmax(dim=1).to("cpu")
        y_true = y.to("cpu")
        correct = pred.eq(y_true)
        for phase in ["train", "valid", "test"]:
            acc[phase] = (
                correct[split_mask[phase]].sum().item()
                / split_mask[phase].sum().item()
            )
    return acc


def train_gnn(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()
    out = model(data.x, data.edge_index)[data.train_mask]
    loss = criterion(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test_gnn(model, data, split_mask, evaluator):
    model.eval()
    out = model(data.x, data.edge_index)
    acc = evaluate(out, data.y, split_mask, evaluator)
    return acc["train"], acc["valid"], acc["test"]


if __name__ == '__main__':
    cfg.merge_from_file('train/configs/cora.yaml')
    cfg = update_cfg(cfg)
    
    run(cfg, train_gnn, test_gnn, train_lm, pretrain_lm, test_lm)
