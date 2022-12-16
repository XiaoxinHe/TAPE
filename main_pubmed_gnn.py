import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from load_pubmed import get_pubmed_casestudy


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.readout = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, readout=True):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if readout:
            x = self.readout(x)
        return x


def train(model, data, optimizer):
    model.train()

    optimizer.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()
    out = model(data.x, data.edge_index)[data.train_mask]
    loss = criterion(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    out = model(data.x, data.edge_index)
    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y)

    train_acc = correct[data.train_mask].sum().item() / \
        data.train_mask.sum().item()
    val_acc = correct[data.val_mask].sum().item() / data.val_mask.sum().item()
    test_acc = correct[data.test_mask].sum().item() / \
        data.test_mask.sum().item()

    return train_acc, val_acc, test_acc, val_loss


def main():

    # load data
    print("[!] Loading dataset")
    lr = 0.001
    data, data_pubid = get_pubmed_casestudy()
    gnn_model = GCN(
        in_channels=data.x.shape[1], hidden_channels=128, out_channels=3, num_layers=4, dropout=0)
    gnn_model.cuda()
    data.cuda()
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=lr,
                                                           patience=20,
                                                           verbose=True)

    best_test_perf = float('-inf')
    for epoch in range(1, 200):
        loss = train(gnn_model, data, optimizer)
        accs = test(gnn_model, data)
        scheduler.step(accs[-1])
        best_test_perf = accs[2] if accs[2] > best_test_perf else best_test_perf
        print(
            f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {accs[0]:.4f}, Val Acc: {accs[1]:.4f}, Test Acc: {best_test_perf:.4f}')


if __name__ == '__main__':
    main()
