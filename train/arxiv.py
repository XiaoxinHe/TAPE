import torch
from core.config import cfg, update_cfg
from core.train_helper import run
from ogb.nodeproppred import Evaluator

BATCH_SIZE = 32


def train_gnn(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()
    out = model(data.x, data.edge_index)[data.train_mask]
    loss = criterion(out, data.y.squeeze(1)[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test_gnn(model, data):
    model.eval()
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


def train_lm(lm, loader, x, optimizer, device):
    cos_sim = torch.nn.CosineSimilarity()
    total_loss = 0
    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        emb_lm = lm(batch)
        gnn_emb = x[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
        loss = (1 - cos_sim(emb_lm, gnn_emb).mean())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


if __name__ == '__main__':
    cfg.merge_from_file('train/configs/arxiv.yaml')
    cfg = update_cfg(cfg)
    evaluator = Evaluator(name='ogbn-arxiv')
    run(cfg, train_gnn, test_gnn, train_lm)
