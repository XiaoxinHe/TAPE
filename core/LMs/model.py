import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from core.LMs.lm_utils import *
from core.utils.function.os_utils import init_random_state, init_path


class BertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, dropout=0.0, seed=0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        self.loss_func = nn.CrossEntropyLoss()

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        init_random_state(seed)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None):

        outputs = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=return_dict,
                                    output_hidden_states=True)
        # outputs[0]=last hidden state
        emb = self.dropout(outputs['hidden_states'][-1])
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        logits = self.classifier(cls_token_emb)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits)


class BertClaInfModel(PreTrainedModel):
    def __init__(self, model, emb, pred, feat_shrink=''):
        super().__init__(model.config)
        self.bert_classifier = model
        self.emb, self.pred = emb, pred
        self.feat_shrink = feat_shrink
        self.loss_func = nn.CrossEntropyLoss()

    @torch.no_grad()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                node_id=None):

        # Extract outputs from the model
        bert_outputs = self.bert_classifier.bert_encoder(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         return_dict=return_dict,
                                                         output_hidden_states=True)
        emb = bert_outputs['hidden_states'][-1]  # outputs[0]=last hidden state

        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.bert_classifier.feat_shrink_layer(
                cls_token_emb)
        logits = self.bert_classifier.classifier(cls_token_emb)

        # Save prediction and embeddings to disk (memmap)
        batch_nodes = node_id.cpu().numpy()
        self.emb[batch_nodes] = cls_token_emb.cpu().numpy().astype(np.float16)
        self.pred[batch_nodes] = logits.cpu().numpy().astype(np.float16)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits)


class ADMMBert(PreTrainedModel):
    def __init__(self, model, n_labels, penalty=0.5, dropout=0.0, seed=0, cla_bias=True, is_augmented=False, feat_shrink='', freeze_bert=False):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        self.ckpt_emb = None
        self.ckpt_pred = None

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert_encoder.parameters():
                param.requires_grad = False

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        init_random_state(seed)
        self.penalty = penalty
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
        if self.ckpt_pred is not None:
            self.ckpt_pred[batch_nodes] = logits.cpu().numpy()

        loss = compute_loss(logits, labels, cls_token_emb, features,
                            gamma, penalty=self.penalty, is_augmented=self.is_augmented)
        return TokenClassifierOutput(loss=loss, logits=logits)


class ADMMBert(PreTrainedModel):
    def __init__(self, model, n_labels, penalty=0.5, dropout=0.0, seed=0, cla_bias=True, is_augmented=False, feat_shrink='', freeze_bert=False):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert_encoder.parameters():
                param.requires_grad = False

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        init_random_state(seed)
        self.penalty = penalty
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

        loss = compute_admm_loss(logits, labels, cls_token_emb, features,
                                 gamma, penalty=self.penalty, is_augmented=self.is_augmented)
        return TokenClassifierOutput(loss=loss, logits=logits)


class InfModel(PreTrainedModel):
    def __init__(self, model, emb, pred, feat_shrink=''):
        super().__init__(model.config)
        self.bert_classifier = model
        self.emb, self.pred = emb, pred
        self.feat_shrink = feat_shrink

    @torch.no_grad()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                features=None,
                node_id=None,
                gamma=None,
                return_dict=None):

        # Extract outputs from the model
        bert_outputs = self.bert_classifier.bert_encoder(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         return_dict=return_dict,
                                                         output_hidden_states=True)

        emb = bert_outputs['hidden_states'][-1]  # outputs[0]=last hidden state
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.bert_classifier.feat_shrink_layer(
                cls_token_emb)
        logits = self.bert_classifier.classifier(cls_token_emb)

        # Save prediction and embeddings to disk (memmap)
        batch_nodes = node_id.cpu().numpy()
        self.emb[batch_nodes] = cls_token_emb.cpu().numpy()
        self.pred[batch_nodes] = logits.cpu().numpy()

        # Output empty to fit the Huggingface trainer pipeline
        empty = torch.zeros((len(node_id), 1)).cuda()
        return TokenClassifierOutput(logits=logits, loss=empty)


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

        loss = compute_kd_loss2(
            cls_token_emb, logits, labels, emb_t, pred_t, pl_weight=self.pl_weight, is_augmented=self.is_augmented)
        return TokenClassifierOutput(loss=loss, logits=logits)
