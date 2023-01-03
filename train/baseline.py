import torch
from core.config import cfg, update_cfg
from core.train_helper import run_baseline as run
from ogb.nodeproppred import Evaluator


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
def test_gnn(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y)
    train_acc = correct[data.train_mask].sum().item() / \
        data.train_mask.sum().item()
    val_acc = correct[data.val_mask].sum().item() / data.val_mask.sum().item()
    test_acc = correct[data.test_mask].sum().item() / \
        data.test_mask.sum().item()
    return train_acc, val_acc, test_acc


def train_arxiv(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()
    out = model(data.x, data.edge_index)[data.train_mask]
    loss = criterion(out, data.y.squeeze(1)[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def test_arxiv(model, data):
    out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[data.train_mask],
        'y_pred': y_pred[data.train_mask],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': data.y[data.val_mask],
        'y_pred': y_pred[data.val_mask],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[data.test_mask],
        'y_pred': y_pred[data.test_mask],
    })['acc']

    return train_acc, val_acc, test_acc


if __name__ == '__main__':
    cfg.merge_from_file('train/configs/cora.yaml')
    cfg = update_cfg(cfg)
    if cfg.dataset == 'ogbn-arxiv':
        evaluator = Evaluator(name='ogbn-arxiv')
        run(cfg, train_arxiv, test_arxiv)
    else:
        run(cfg, train_gnn, test_gnn)
