import numpy as np


def compute_metrics(p):
    from sklearn.metrics import accuracy_score
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy}


def compute_loss(logits, labels, emb, pesudo_emb, pl_weight=0.5, is_augmented=False):
    """
    Combine two types of losses: (1-α)*MLE (CE loss on gold) + α*Pl_loss (CE loss on pseudo labels)
    """
    import torch as th
    cross_entropy = th.nn.CrossEntropyLoss()
    cos_sim = th.nn.CosineSimilarity()

    if is_augmented:
        # def deal_nan(x): return 0 if th.isnan(x) else x
        # mle_loss = deal_nan(cross_entropy(logits, labels))
        pl_loss = (1-cos_sim(emb, pesudo_emb)).mean()
        loss = pl_loss
        # loss = pl_weight * pl_loss + (1 - pl_weight) * mle_loss
        # print(mle_loss, pl_loss)
    else:
        loss = cross_entropy(logits, labels)
    return loss
