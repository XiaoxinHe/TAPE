import torch.nn as nn
import torch.nn.functional as F
import core.model_utils.pyg_gnn_wrapper as gnn_wrapper


class GNN(nn.Module):
    def __init__(self,
                 nhid,
                 nout,
                 nlayer,
                 gnn_type,
                 dropout=0.0,
                 res=True):
        super().__init__()
        self.dropout = dropout
        self.res = res
        self.nlayer = nlayer
        self.convs = nn.ModuleList(
            [getattr(gnn_wrapper, gnn_type)(nhid, nhid) for _ in range(nlayer)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(nhid)
                                   for _ in range(nlayer)])
        self.output_encoder = nn.Linear(nhid, nout)

    def forward(self, x, edge_index, readout=True):
        previous_x = x
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            # if self.res:
            #     x = x + previous_x
            #     previous_x = x
        if readout:
            x = self.output_encoder(x)
        return x
