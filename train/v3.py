import torch
from core.config import cfg, update_cfg
from core.train_helper import run_v2 as run

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
    loss0 = criterion(out, data.y[data.train_mask])
    loss1 = alpha * (1 - cos_sim(z, lm_z).mean())
    loss = loss0 + loss1
    loss.backward()
    optimizer.step()
    optimizer_z.step()
    return loss.item(), loss0.item(), loss1.item()


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
        out = model(batch, readout=True)
        loss = criterion(out[batch_train_mask], batch_y[batch_train_mask])
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
        out = model(batch, readout=True)
        outs.append(out.cpu())
    out = torch.cat(outs, dim=0)
    acc = evaluate(out, data.y, split_mask, evaluator)
    return acc["train"], acc["valid"], acc["test"]


def train_lm(lm, loader, z, optimizer, device):
    cos_sim = torch.nn.CosineSimilarity()
    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        emb_lm = lm(batch)
        z_ = z[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
        loss = (1 - cos_sim(emb_lm, z_).mean())
        loss.backward()
        optimizer.step()
    return loss


if __name__ == '__main__':
    cfg.merge_from_file('train/configs/cora.yaml')
    cfg = update_cfg(cfg)
    run(cfg, train_gnn, test_gnn, train_lm, pretrain_lm, test_lm)
