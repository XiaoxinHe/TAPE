from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
import core.model_utils.pyg_gnn_wrapper as gnn_wrapper


# class GNN(nn.Module):
#     def __init__(self,
#                  nhid,
#                  nout,
#                  nlayer,
#                  gnn_type,
#                  dropout=0.0,
#                  res=True):
#         super().__init__()
#         self.dropout = dropout
#         self.res = res
#         self.nlayer = nlayer
#         self.convs = nn.ModuleList(
#             [getattr(gnn_wrapper, gnn_type)(nhid, nhid) for _ in range(nlayer)])
#         self.norms = nn.ModuleList([nn.BatchNorm1d(nhid)
#                                    for _ in range(nlayer)])
#         self.output_encoder = nn.Linear(nhid, nout)

#     def forward(self, x, edge_index, readout=True):
#         previous_x = x
#         for conv, norm in zip(self.convs, self.norms):
#             x = conv(x, edge_index)
#             x = norm(x)
#             x = F.relu(x)
#             x = F.dropout(x, self.dropout, training=self.training)
#             # if self.res:
#             #     x = x + previous_x
#             #     previous_x = x
#         if readout:
#             x = self.output_encoder(x)
#         return x


class GCN(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout=0.0):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
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
