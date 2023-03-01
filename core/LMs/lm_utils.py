import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def compute_metrics(p):
    from sklearn.metrics import accuracy_score
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy}


def compute_loss(logits, labels, emb, pesudo_emb, pl_weight=0.5, is_augmented=False):
    cross_entropy = nn.CrossEntropyLoss()
    cos_sim = nn.CosineSimilarity()

    if is_augmented:
        # def deal_nan(x): return 0 if th.isnan(x) else x
        # mle_loss = deal_nan(cross_entropy(logits, labels))
        pl_loss = (1-cos_sim(emb, pesudo_emb)).sum()
        loss = pl_loss
        # loss = pl_weight * pl_loss + (1 - pl_weight) * mle_loss
        # print(mle_loss.item(), pl_loss.item())
    else:
        def deal_nan(x): return 0 if torch.isnan(x) else x
        # print(logits.shape, labels.shape)
        loss = deal_nan(cross_entropy(logits, labels))
    return loss


def compute_admm_loss(logits, labels, emb, pesudo_emb, gamma, penalty=0.5, is_augmented=False):

    if is_augmented:
        l2_loss = torch.nn.MSELoss()
        loss = 0.5*penalty*l2_loss(emb, pesudo_emb+gamma/penalty)
        # loss = l2_loss(emb, pesudo_emb+gamma/penalty)
    else:
        cross_entropy = torch.nn.CrossEntropyLoss()
        def deal_nan(x): return 0 if torch.isnan(x) else x
        loss = deal_nan(cross_entropy(logits, labels))
    return loss


def compute_kd_loss(emb, pred, labels, emb_t, pred_t, pl_weight=0.5, is_augmented=False, T=1):
    if is_augmented:
        hard_loss = F.cross_entropy(pred, labels) * (1. - pl_weight)
        dis_loss = nn.KLDivLoss()(F.log_softmax(pred/T, dim=1),
                                  F.softmax(pred_t/T, dim=1)) * (pl_weight * T * T)

        cos_loss = (1 - nn.CosineSimilarity(dim=-1)
                    (emb, emb_t)).mean() * pl_weight
        # print(hard_loss.item(), soft_loss.item())
        loss = hard_loss + dis_loss + cos_loss
    else:
        def deal_nan(x): return 0 if torch.isnan(x) else x
        criterion = torch.nn.CrossEntropyLoss()
        loss = deal_nan(criterion(pred, labels))

    return loss


def load_data(dataset, use_text=False):

    if dataset == 'cora':
        from core.data_utils.load_cora import get_raw_text_cora as get_raw_text
    elif dataset == 'pubmed':
        from core.data_utils.load_pubmed import get_raw_text_pubmed as get_raw_text
    elif dataset == 'citeseer':
        from core.data_utils.load_citeseer import get_raw_text_citeseer as get_raw_text
    elif dataset == 'ogbn-arxiv':
        from core.data_utils.load_arxiv import get_raw_text_arxiv as get_raw_text
    elif dataset == 'ogbn-products':
        from core.data_utils.load_products import get_raw_text_products as get_raw_text

    data, text = get_raw_text(use_text)

    return data, text
