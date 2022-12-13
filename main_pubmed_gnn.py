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
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
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

    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y)

    train_acc = correct[data.train_mask].sum().item() / \
        data.train_mask.sum().item()
    val_acc = correct[data.val_mask].sum().item() / data.val_mask.sum().item()
    test_acc = correct[data.test_mask].sum().item() / \
        data.test_mask.sum().item()

    return train_acc, val_acc, test_acc


def main():

    # load data
    print("[!] Loading dataset")

    data, data_pubid = get_pubmed_casestudy()
    gnn_model = GCN(
        in_channels=data.x.shape[1], hidden_channels=128, out_channels=3, num_layers=4, dropout=0)
    gnn_model.cuda()

    data.cuda()

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    for epoch in range(1, 1000):
        loss = train(gnn_model, data, optimizer)
        accs = test(gnn_model, data)
        print(
            f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {accs[0]:.4f}, Val Acc: {accs[1]:.4f}, Test Acc: {accs[2]:.4f}')


if __name__ == '__main__':
    main()
