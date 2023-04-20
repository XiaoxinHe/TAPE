from torch_geometric.nn import GCNConv, SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SAGE(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout=0.0,
                 activation=F.relu,
                 norm='BN',
                 input_norm=False):
        super(SAGE, self).__init__()

        if input_norm:
            self.input_norm = nn.BatchNorm1d(in_channels)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, "mean"))

        norm_layer = nn.BatchNorm1d if norm == 'BN' else nn.LayerNorm
        self.norms = nn.ModuleList()
        self.norms.append(norm_layer(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, "mean"))
            self.norms.append(norm_layer(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels, "mean"))

        self.dropout = dropout
        self.activation = activation

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        if hasattr(self, 'input_norm'):
            x = self.input_norm(x)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.norms[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
