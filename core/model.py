import imp
from turtle import forward
from torch_geometric.nn import GCNConv
from transformers import PreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


# class BertClassifier(PreTrainedModel):

#     def __init__(self, feat_shrink, nout=-1):
#         model = BertModel.from_pretrained(
#             'bert-base-uncased',
#             output_attentions=False,
#             output_hidden_states=True,
#         )
#         super().__init__(model.config)
#         self.bert_encoder = model
#         self.feat_shrink_layer = torch.nn.Linear(768, feat_shrink)

#         # Instantiate an one-layer feed-forward classifier
#         if nout > 0:
#             self.classifier = nn.Sequential(
#                 nn.Linear(feat_shrink, feat_shrink),
#                 nn.ReLU(),
#                 # nn.Dropout(0.5),
#                 nn.Linear(feat_shrink, nout)
#             )

#     def forward(self, batch):
#         b_input_ids, b_input_mask = batch
#         output = self.bert_encoder(b_input_ids,
#                                    token_type_ids=None,
#                                    attention_mask=b_input_mask,
#                                    output_hidden_states=True)
#         emb = output['hidden_states'][-1]  # outputs[0]=last hidden state
#         cls_token_emb = emb.permute(1, 0, 2)[0]
#         cls_token_emb = self.feat_shrink_layer(cls_token_emb)
#         return cls_token_emb

#     def classif(self, cls_token_emb):
#         return self.classifier(cls_token_emb)

#     def generate_node_features(self, loader, device):
#         features = []
#         for batch in loader:
#             batch = tuple(t.to(device) for t in batch)
#             output = self.forward(batch)
#             features.append(output.detach().cpu())
#         features = torch.cat(features, dim=0)
#         return features


class BertClassifier(PreTrainedModel):

    def __init__(self, feat_shrink, nout=-1):
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
