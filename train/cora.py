import torch
from core.config import cfg, update_cfg
from core.model import GCN
from core.train_helper import run

BATCH_SIZE = 64


def train_gnn_epoch(model, data, optimizer):
    optimizer.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()
    out = model(data.x, data.edge_index)[data.train_mask]
    loss = criterion(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def train_gnn(data, model, optimizer):
    best_val_perf = best_test_perf = float('-inf')
    for epoch in range(1, cfg.train.epochs+1):
        model.train()
        loss = train_gnn_epoch(model, data, optimizer)
        model.eval()
        train_acc, val_acc, test_acc = test_gnn(model, data)
        if val_acc > best_val_perf:
            best_train_acc = train_acc
            best_val_perf = val_acc
            best_test_perf = test_acc
            best_model = model
        if epoch % 10 == 0:
            print(
                f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {best_test_perf:.4f}')

    x = best_model(data.x, data.edge_index, False).detach()
    return best_train_acc, best_val_perf, best_test_perf, x


def train_lm(lm, loader, data, optimizer, device):
    cos_sim = torch.nn.CosineSimilarity()
    total_loss = 0
    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        emb_lm = lm(batch)
        gnn_emb = data.x[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
        loss = (1 - cos_sim(emb_lm, gnn_emb).mean())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


@torch.no_grad()
def test_gnn(model, data):
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y)

    train_acc = correct[data.train_mask].sum().item() / \
        data.train_mask.sum().item()
    val_acc = correct[data.val_mask].sum().item() / data.val_mask.sum().item()
    test_acc = correct[data.test_mask].sum().item() / \
        data.test_mask.sum().item()

    return train_acc, val_acc, test_acc


if __name__ == '__main__':
    cfg.merge_from_file('train/configs/cora.yaml')
    cfg = update_cfg(cfg)
    run(cfg, train_gnn, train_lm)
