import torch
from core.config import cfg, update_cfg
from core.model import GCN
from core.train_helper import run

BATCH_SIZE = 32


def train_gnn_epoch(model, data, optimizer):
    model.train()

    optimizer.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()
    out = model(data.x, data.edge_index)[data.train_mask]
    loss = criterion(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


def train_gnn(cfg, data):
    model = GCN(in_channels=768, hidden_channels=128,
                out_channels=7, num_layers=4, dropout=0)
    model = model.to(cfg.device)
    data = data.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr_gnn)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5,
                                                           patience=cfg.train.lr_patience,
                                                           verbose=False)
    best_val_perf = best_test_perf = float('-inf')
    best_model = model
    for epoch in range(1, cfg.train.epochs_gnn+1):
        model.train()
        loss = train_gnn_epoch(model, data, optimizer)
        accs = test(model, data)
        scheduler.step(accs[-1])
        if accs[1] > best_val_perf:
            best_val_perf = accs[1]
            best_test_perf = accs[2]
            best_model = model
        if epoch % 10 == 0:
            print(
                f'InnerEpoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {accs[0]:.4f}, Val Acc: {accs[1]:.4f}, Test Acc: {best_test_perf:.4f}')

    features = best_model(data.x, data.edge_index, False).detach()
    return features, best_test_perf


def train_lm(lm, loader, features, optimizer, device):
    criterion = torch.nn.CosineSimilarity()
    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        emb_lm = lm(batch, shrink=True)
        gnn_emb = features[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
        loss = (1-criterion(emb_lm, gnn_emb).mean())
        loss.backward()
        optimizer.step()

    features = []
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        output = lm(batch)
        features.append(output.detach().cpu())
    features = torch.cat(features, dim=0)
    return features


@torch.no_grad()
def test(model, data):
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


if __name__ == '__main__':
    cfg.merge_from_file('train/configs/cora.yaml')
    cfg = update_cfg(cfg)
    run(cfg, train_gnn, train_lm)
