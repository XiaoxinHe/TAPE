import torch
from core.config import cfg, update_cfg
from core.train_helper import run_v2 as run
from ogb.nodeproppred import Evaluator

BATCH_SIZE = 32



def train_gnn(model, model_z, data, lm_z, optimizer, optimizer_z):
    criterion = torch.nn.CrossEntropyLoss()
    cos_sim = torch.nn.CosineSimilarity()
    model.train()
    model_z.train()
    optimizer.zero_grad()
    optimizer_z.zero_grad()
    z = model_z()
    out = model(z, data.edge_index)[data.train_mask]
    loss0 = criterion(out, data.y.squeeze(1)[data.train_mask])
    loss1 = 0.01*(1 - cos_sim(z, lm_z).mean())
    loss = loss0 + loss1
    loss.backward()
    optimizer.step()
    optimizer_z.step()
    return loss.item(), loss0.item(), loss1.item()

@torch.no_grad()
def test_gnn(model, model_z, data):
    model.eval()
    model_z.eval()
    z = model_z()
    out = model(z, data.edge_index)
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


def pretrain_lm(model, loader, data, optimizer, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        batch_train_mask = data.train_mask[BATCH_SIZE *
                                           batch_idx: BATCH_SIZE*(batch_idx+1)]
        batch_val_mask = data.val_mask[BATCH_SIZE *
                                       batch_idx: BATCH_SIZE*(batch_idx+1)]
        batch_test_mask = data.test_mask[BATCH_SIZE *
                                         batch_idx: BATCH_SIZE*(batch_idx+1)]
        batch_y = data.y[BATCH_SIZE*batch_idx: BATCH_SIZE * (batch_idx+1)]
        out = model(batch, readout=True)
        y_pred = out.argmax(dim=-1, keepdim=True)
        loss = criterion(out[batch_train_mask],
                         batch_y.squeeze(1)[batch_train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        try:
            train_acc = evaluator.eval({
                'y_true': batch_y[batch_train_mask],
                'y_pred': y_pred[batch_train_mask],
            })['acc']
        except:
            train_acc = 0
        try:
            val_acc = evaluator.eval({
                'y_true': batch_y[batch_val_mask],
                'y_pred': y_pred[batch_val_mask],
            })['acc']
        except:
            val_acc = 0
        try:
            test_acc = evaluator.eval({
                'y_true': batch_y[batch_test_mask],
                'y_pred': y_pred[batch_test_mask],
            })['acc']
        except:
            test_acc = 0

        print(f'Step: {batch_idx:02d}, Train Loss: {loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

    return total_loss/len(loader)


@ torch.no_grad()
def test_lm(model, loader, data, device):
    model.eval()

    train_accs = []
    val_accs = []
    test_accs = []
    for batch_idx, batch in enumerate(loader):
        batch_train_mask = data.train_mask[BATCH_SIZE *
                                           batch_idx: BATCH_SIZE*(batch_idx+1)]
        batch_val_mask = data.val_mask[BATCH_SIZE *
                                       batch_idx: BATCH_SIZE*(batch_idx+1)]
        batch_test_mask = data.test_mask[BATCH_SIZE *
                                         batch_idx: BATCH_SIZE*(batch_idx+1)]
        batch_y = data.y[BATCH_SIZE * batch_idx: BATCH_SIZE*(batch_idx+1)]
        batch = tuple(t.to(device) for t in batch)

        out = model(batch, readout=True)
        y_pred = out.argmax(dim=-1, keepdim=True)

        try:
            train_acc = evaluator.eval({
                'y_true': batch_y[batch_train_mask],
                'y_pred': y_pred[batch_train_mask],
            })['acc']
        except:
            train_acc = 0
        try:
            val_acc = evaluator.eval({
                'y_true': batch_y[batch_val_mask],
                'y_pred': y_pred[batch_val_mask],
            })['acc']
        except:
            val_acc = 0
        try:
            test_acc = evaluator.eval({
                'y_true': batch_y[batch_test_mask],
                'y_pred': y_pred[batch_test_mask],
            })['acc']
        except:
            test_acc = 0

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    train_acc = sum(train_accs)/len(train_accs)
    val_acc = sum(val_accs)/len(val_accs)
    test_acc = sum(test_accs)/len(test_accs)
    return train_acc, val_acc, test_acc


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
    cfg.merge_from_file('train/configs/arxiv.yaml')
    cfg = update_cfg(cfg)
    evaluator = Evaluator(name='ogbn-arxiv')
    run(cfg, train_gnn, test_gnn, train_lm, pretrain_lm, test_lm)
