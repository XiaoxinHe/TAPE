import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from core.LMs.lm_utils import *
from core.utils.function.os_utils import init_random_state, init_path


class BertEmb(PreTrainedModel):
    def __init__(self, model, n_labels, pseudo_label_weight=0.5, dropout=0.0, seed=0, cla_bias=True, is_augmented=False, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        self.ckpt_emb = None

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        init_random_state(seed)
        self.pl_weight = pseudo_label_weight
        self.is_augmented = is_augmented

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                features=None,
                node_id=None,
                return_dict=None):
        outputs = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=return_dict,
                                    output_hidden_states=True)
        emb = self.dropout(outputs['hidden_states'][-1])
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        logits = self.classifier(cls_token_emb)
        if labels.shape[-1] == 1:
            labels = labels.squeeze()

        batch_nodes = node_id.cpu().numpy()
        if self.ckpt_emb is not None:
            self.ckpt_emb[batch_nodes] = cls_token_emb.cpu().numpy()
        loss = compute_loss(logits, labels, cls_token_emb, features,
                            pl_weight=self.pl_weight, is_augmented=self.is_augmented)
        return TokenClassifierOutput(loss=loss, logits=logits)


class ADMMBert(PreTrainedModel):
    def __init__(self, model, n_labels, pseudo_label_weight=0.5, dropout=0.0, seed=0, cla_bias=True, is_augmented=False, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        self.ckpt_emb = None

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        init_random_state(seed)
        self.pl_weight = pseudo_label_weight
        self.is_augmented = is_augmented

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                features=None,
                node_id=None,
                gamma=None,
                return_dict=None):
        outputs = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=return_dict,
                                    output_hidden_states=True)
        emb = self.dropout(outputs['hidden_states'][-1])
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        logits = self.classifier(cls_token_emb)
        if labels.shape[-1] == 1:
            labels = labels.squeeze()

        batch_nodes = node_id.cpu().numpy()
        if self.ckpt_emb is not None:
            self.ckpt_emb[batch_nodes] = cls_token_emb.cpu().numpy()
        loss = compute_admm_loss(logits, labels, cls_token_emb, features,
                                 gamma, penalty=self.pl_weight, is_augmented=self.is_augmented)
        return TokenClassifierOutput(loss=loss, logits=logits)


# class ADMMBertInf(PreTrainedModel):
#     def __init__(self, model, ckpt_emb):
#         super().__init__(model.config)
#         self.model = model
#         self.ckpt_emb = ckpt_emb
#         self.feat_shrink = self.model.feat_shrink
#         self.pl_weight = self.model.pl_weight
#         self.is_augmented = self.model.is_augmented

#     def forward(self,
#                 input_ids=None,
#                 attention_mask=None,
#                 labels=None,
#                 features=None,
#                 node_id=None,
#                 gamma=None,
#                 return_dict=None):
#         outputs = self.model.bert_encoder(input_ids=input_ids,
#                                           attention_mask=attention_mask,
#                                           return_dict=return_dict,
#                                           output_hidden_states=True)
#         emb = outputs['hidden_states'][-1]
#         cls_token_emb = emb.permute(1, 0, 2)[0]
#         if self.feat_shrink:
#             cls_token_emb = self.model.feat_shrink_layer(cls_token_emb)
#         logits = self.model.classifier(cls_token_emb)
#         if labels.shape[-1] == 1:
#             labels = labels.squeeze()
#         loss = compute_admm_loss(logits, labels, cls_token_emb, features,
#                                  gamma, penalty=self.pl_weight, is_augmented=self.is_augmented)

#         batch_nodes = node_id.cpu().numpy()
#         self.ckpt_emb[batch_nodes] = cls_token_emb.cpu().numpy()

#         return TokenClassifierOutput(loss=loss, logits=logits)


class KDBert(PreTrainedModel):
    def __init__(self, model, n_labels, pseudo_label_weight=0.5, dropout=0.0, seed=0, cla_bias=True, is_augmented=False, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        self.ckpt_emb = None
        self.ckpt_pred = None
        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        init_random_state(seed)
        self.pl_weight = pseudo_label_weight
        self.is_augmented = is_augmented

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                emb_t=None,
                pred_t=None,
                node_id=None,
                return_dict=None,
                token_type_ids=None):
        outputs = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=return_dict,
                                    output_hidden_states=True,
                                    token_type_ids=token_type_ids)
        emb = self.dropout(outputs['hidden_states'][-1])
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        logits = self.classifier(cls_token_emb)
        if labels.shape[-1] == 1:
            labels = labels.squeeze()

        batch_nodes = node_id.cpu().numpy()
        if self.ckpt_emb is not None:
            self.ckpt_emb[batch_nodes] = cls_token_emb.cpu().numpy()
        if self.ckpt_pred is not None:
            self.ckpt_pred[batch_nodes] = logits.cpu().numpy()

        loss = compute_kd_loss2(cls_token_emb, logits,  labels, emb_t,
                                pred_t, pl_weight=self.pl_weight, is_augmented=self.is_augmented)
        return TokenClassifierOutput(loss=loss, logits=logits)
