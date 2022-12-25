import imp
from turtle import forward
from torch_geometric.nn import GCNConv
from transformers import PreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertClassifier(PreTrainedModel):

    def __init__(self, feat_shrink):
        model = BertModel.from_pretrained(
            'bert-base-uncased',
            output_attentions=False,
            output_hidden_states=True,
        )
        super().__init__(model.config)
        self.bert_encoder = model
        self.feat_shrink_layer = torch.nn.Linear(768, feat_shrink)

    def forward(self, batch):
        b_input_ids, b_input_mask = batch
        output = self.bert_encoder(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   output_hidden_states=True)
        emb = output['hidden_states'][-1]  # outputs[0]=last hidden state
        cls_token_emb = emb.permute(1, 0, 2)[0]
        cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        return cls_token_emb

    def generate_node_features(self, loader, device):
        features = []
        for batch in loader:
            batch = tuple(t.to(device) for t in batch)
            output = self.forward(batch)
            features.append(output.detach().cpu())
        features = torch.cat(features, dim=0)
        return features


class Z(torch.nn.Module):
    def __init__(self, z):
        super(Z, self).__init__()
        self.Z = nn.Parameter(z)

    def forward(self):
        return self.Z


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
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.readout = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, readout=True):
        for conv, norm in zip(self.convs, self.bns):
            x = conv(x, adj_t)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if readout:
            x = self.readout(x)
        return x
