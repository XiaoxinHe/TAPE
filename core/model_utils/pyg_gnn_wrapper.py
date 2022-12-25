import torch.nn as nn
import torch_geometric.nn as gnn
from core.model_utils.elements import MLP


class GCNConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        # self.nn = MLP(nin, nout, 2, False, bias=bias)
        # self.layer = gnn.GCNConv(nin, nin, bias=True)
        self.layer = gnn.GCNConv(nin, nout, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index):
        return self.layer(x, edge_index)
        # return self.nn(F.relu(self.layer(x, edge_index)))


class ResGatedGraphConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.layer = gnn.ResGatedGraphConv(nin, nout, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index):
        return self.layer(x, edge_index)


class GINEConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.nn = MLP(nin, nout, 2, False, bias=bias)
        self.layer = gnn.GINEConv(self.nn, train_eps=True)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index):
        return self.layer(x, edge_index)


class TransformerConv(nn.Module):
    def __init__(self, nin, nout, bias=True, nhead=8):
        super().__init__()
        self.layer = gnn.TransformerConv(
            in_channels=nin, out_channels=nout//nhead, heads=nhead, edge_dim=nin, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index):
        return self.layer(x, edge_index)


class GATConv(nn.Module):
    def __init__(self, nin, nout, bias=True, nhead=1):
        super().__init__()
        self.layer = gnn.GATConv(nin, nout//nhead, nhead, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index):
        return self.layer(x, edge_index)


class GatedGraphConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.layer = gnn.GatedGraphConv(nin, nout, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index):
        return self.layer(x, edge_index)
