import torch
from core.config import cfg, update_cfg
from core.train_helper import run
import os

BATCH_SIZE = 32


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


def train_gnn(model, model_z, data, lm_z, optimizer, optimizer_z, alpha):
    criterion = torch.nn.CrossEntropyLoss()
    cos_sim = torch.nn.CosineSimilarity()
    model.train()
    model_z.train()
    optimizer.zero_grad()
    optimizer_z.zero_grad()
    z = model_z()
    out = model(z, data.edge_index)[data.train_mask]
    loss0 = alpha * criterion(out, data.y[data.train_mask])
    loss1 = (1-alpha) * (1 - cos_sim(z, lm_z).mean())
    loss = loss0 + loss1
    loss.backward()
    optimizer.step()
    optimizer_z.step()
    return loss.item(), loss0.item(), loss1.item()


def train_lm(lm, loader, z, data, optimizer, beta, split_mask, evaluator, device, path):
    best_val = 0
    criterion = torch.nn.CrossEntropyLoss()
    cos_sim = torch.nn.CosineSimilarity()
    total_loss = 0
    total_loss0 = 0
    total_loss1 = 0
    eval_steps = 5

    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        z_ = z[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
        y = data.y[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
        emb, pred = lm(batch)
        train_mask = data.train_mask[batch_idx *
                                     BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
        loss0 = beta * criterion(pred[train_mask], y[train_mask])
        loss1 = (1-beta) * (1 - cos_sim(emb, z_).mean())
        loss = loss0 + loss1
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_loss0 += loss0.item()
        total_loss1 += loss1.item()
        if batch_idx % eval_steps == 0:
            train_acc, val_acc, test_acc = test_lm(
                lm, loader, data, split_mask, evaluator, device)
            print(f'batch_idx: {batch_idx}, Loss: {loss.item():.4f}, Loss0: {loss0.item():.4f}, Loss1: {loss1.item():.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
            if val_acc > best_val:
                torch.save(lm.state_dict(), path)
    return total_loss/len(loader), total_loss0/len(loader), total_loss1/len(loader)


@torch.no_grad()
def test_gnn(model, model_z, data, split_mask, evaluator):
    model.eval()
    model_z.eval()
    z = model_z()
    out = model(z, data.edge_index)
    acc = evaluate(out, data.y, split_mask, evaluator)
    return acc["train"], acc["valid"], acc["test"]


def pretrain_lm(model, loader, data, optimizer, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    total_loss = 0
    data = data.to(device)
    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        batch_train_mask = data.train_mask[BATCH_SIZE *
                                           batch_idx: BATCH_SIZE*(batch_idx+1)]
        batch_y = data.y[BATCH_SIZE*batch_idx: BATCH_SIZE * (batch_idx+1)]
        emb, pred = model(batch)
        loss = criterion(pred[batch_train_mask], batch_y[batch_train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader)


@ torch.no_grad()
def test_lm(model, loader, data, split_mask, evaluator, device):
    model.eval()
    outs = []
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        emb, pred = model(batch)
        outs.append(pred.cpu())
    out = torch.cat(outs, dim=0)
    acc = evaluate(out, data.y, split_mask, evaluator)
    return acc["train"], acc["valid"], acc["test"]


if __name__ == '__main__':
    cfg.merge_from_file('train/configs/cora.yaml')
    cfg = update_cfg(cfg)
    run(cfg, train_gnn, test_gnn, train_lm, pretrain_lm, test_lm)
