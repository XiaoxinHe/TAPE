import torch
from core.config import cfg, update_cfg
from core.train_helper import run_baseline_lm as run

BATCH_SIZE = 32


def train(model, loader, data, optimizer, device):
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
        out = model(batch)
        pred = out.argmax(dim=-1)
        correct = pred.eq(batch_y)
        
        train_acc = correct[batch_train_mask].sum(
        ).item() / batch_train_mask.sum().item()
        val_acc = correct[batch_val_mask].sum().item() / \
            batch_val_mask.sum().item()
        test_acc = correct[batch_test_mask].sum().item() / \
            batch_test_mask.sum().item()

        loss = criterion(out[batch_train_mask], batch_y[batch_train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # print(f'Step: {batch_idx:02d}, Train Loss: {loss:.4f}, '
        #       f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
    return total_loss/len(loader)


@ torch.no_grad()
def test(model, loader, data, device):
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

        out = model(batch)
        pred = out.argmax(dim=-1)
        correct = pred.eq(batch_y)
        train_acc = correct[batch_train_mask].sum(
        ).item() / batch_train_mask.sum().item()
        val_acc = correct[batch_val_mask].sum().item() / \
            batch_val_mask.sum().item()
        test_acc = correct[batch_test_mask].sum().item() / \
            batch_test_mask.sum().item()

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    train_acc = sum(train_accs)/len(train_accs)
    val_acc = sum(val_accs)/len(val_accs)
    test_acc = sum(test_accs)/len(test_accs)
    return train_acc, val_acc, test_acc


if __name__ == '__main__':
    cfg.merge_from_file('train/configs/cora.yaml')
    cfg = update_cfg(cfg)
    run(cfg, train, test)
