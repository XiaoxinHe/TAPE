import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import json


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
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(emb, pesudo_emb+gamma/penalty)
        # tmp = pesudo_emb-emb
        # loss = (gamma*tmp).mean() + 0.5*penalty*((tmp**2).mean())
    else:
        cross_entropy = torch.nn.CrossEntropyLoss()
        loss = cross_entropy(logits, labels)
    return loss


def compute_kd_loss(emb, pred, labels, emb_t, pred_t, pl_weight=0.5, is_augmented=False, T=1):
    cross_entropy = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    if is_augmented:
        hard_loss = cross_entropy(pred, labels) * (1. - pl_weight)
        dis_loss = nn.KLDivLoss()(F.log_softmax(pred/T, dim=1),
                                  F.softmax(pred_t/T, dim=1)) * (pl_weight * T * T)

        cos_loss = (1 - nn.CosineSimilarity(dim=-1)
                    (emb, emb_t)).mean() * pl_weight
        # print(hard_loss.item(), soft_loss.item())
        loss = hard_loss + dis_loss + cos_loss
    else:
        loss = cross_entropy(pred, labels)

    return loss


def compute_kd_loss2(emb, pred, labels, emb_t, pred_t, pl_weight=0.5, is_augmented=False):
    if is_augmented:
        hard_loss = F.cross_entropy(pred, labels)
        # sim = F.softmax(torch.matmul(emb, emb.T), dim=-1)
        # sim_t = F.softmax(torch.matmul(emb_t, emb_t.T), dim=-1)
        # sim = torch.matmul(emb, emb.T)
        # sim_t = torch.matmul(emb_t, emb_t.T)
        # loss_relative_sim = torch.mean((sim-sim_t)**2)

        loss_soft_label = nn.KLDivLoss(reduction="batchmean", log_target=True)(
            pred.log_softmax(dim=1), pred_t.log_softmax(dim=1))
        loss_relative_sim = (
            1 - nn.CosineSimilarity(dim=-1)(emb, emb_t)).mean()

        # print(hard_loss.item(), loss_soft_label.item(), loss_relative_sim.item())
        # return hard_loss + loss_soft_label + loss_relative_sim
        # print(hard_loss.item(), loss_soft_label.item())
        return hard_loss+loss_soft_label+loss_relative_sim

    else:
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(pred, labels)
        return loss


def load_data(dataset, use_text=False, use_gpt=False):

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

    if use_gpt:
        dataset, text = get_raw_text(False)
        data = dataset[0]
        folder_path = 'gpt_responses/{}'.format(dataset)
        print(f"using gpt: {folder_path}")
        n = data.y.shape[0]
        text = []
        for i in range(n):
            filename = str(i) + '.json'
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                text.append(json_data['choices'][0]['message']['content'])
    else:
        dataset, text = get_raw_text(use_text)

    return dataset, text
