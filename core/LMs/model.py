import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from core.LMs.lm_utils import compute_loss
from core.utils.function.os_utils import init_random_state, init_path


class BertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, pseudo_label_weight=0.5, dropout=0.0, seed=0, cla_bias=True, is_augmented=False, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
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
                return_dict=None,):
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
        pesudo_emb = None
        loss = compute_loss(logits, labels, emb, pesudo_emb,
                            pl_weight=self.pl_weight, is_augmented=self.is_augmented)
        return TokenClassifierOutput(loss=loss, logits=logits)


class BertClaInfModel(PreTrainedModel):
    def __init__(self, model):
        super().__init__(model.config)
        self.bert_classifier = model
        n_nodes = 2708
        self.emb = np.memmap(init_path("output/bert.emb"), dtype=np.float16, mode='w+',
                             shape=(n_nodes, 768))
        self.pred = np.memmap(init_path("output/bert.pred"), dtype=np.float16, mode='w+',
                              shape=(n_nodes, 7))

    @torch.no_grad()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                node_id=None):
        bert_outputs = self.bert_classifier.bert_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            output_hidden_states=True)
        emb = bert_outputs['hidden_states'][-1]
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.bert_classifier.feat_shrink:
            cls_token_emb = self.bert_classifier.feat_shrink_layer(
                cls_token_emb)
        logits = self.bert_classifier.classifier(cls_token_emb)

        # Save prediction and embeddings to disk (memmap)
        batch_nodes = node_id.cpu().numpy()

        self.emb[batch_nodes] = cls_token_emb.cpu().numpy().astype(np.float16)
        self.pred[batch_nodes] = logits.cpu().numpy().astype(np.float16)

        pesudo_emb = None
        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = compute_loss(logits, labels, emb, pesudo_emb, pl_weight=self.bert_classifier.pl_weight,
                            is_augmented=self.bert_classifier.is_augmented)
        return TokenClassifierOutput(logits=logits, loss=loss)
