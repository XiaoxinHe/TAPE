import numpy as np
from core.utils.modules.conf_utils import ModelConfig


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
        def deal_nan(x): return 0 if th.isnan(x) else x
        mle_loss = deal_nan(cross_entropy(logits, labels))
        pl_loss = deal_nan(cos_sim(emb, pesudo_emb))
        loss = pl_weight * pl_loss + (1 - pl_weight) * mle_loss
    else:
        loss = cross_entropy(logits, labels)
    return loss


class LMConfig(ModelConfig):
    def __init__(self, args=None):
        # ! INITIALIZE ARGS
        super(LMConfig, self).__init__('LMs')

        # ! LM Settings
        self.model = 'Bert'
        self.init_ckpt = 'Prt'

        self.lr = 0.00002
        self.eq_batch_size = 36
        self.weight_decay = 0.01
        self.label_smoothing_factor = 0.1
        self.dropout = 0.1
        self.warmup_epochs = 0.2
        self.att_dropout = 0.1
        self.cla_dropout = 0.1
        self.cla_bias = 'T'
        self.grad_acc_steps = 2
        self.load_best_model_at_end = 'T'

        # ! glem Training settings
        self.is_augmented = False
        self.save_folder = ''
        self.emi_file = ''
        self.em_iter = 0
        self.ce_reduction = 'mean'

        # self.feat_shrink = '100'
        self.feat_shrink = ''
        self.pl_weight = 0.5  # pseudo_label_weight
        self.pl_ratio = 0.5  # pseudo_label data ratio
        self.eval_patience = 100000
        self.is_inf = False

        # to be called in the instances of LMConfig.
