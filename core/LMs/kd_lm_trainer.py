import torch
import numpy as np
from core.utils.data.dataset import KDDataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, IntervalStrategy
from core.LMs.model import KDBert, InfModel

from core.LMs.lm_utils import load_data
from core.LMs.lm_utils import compute_metrics
from core.utils.function.os_utils import init_path, time_logger

feat_shrink = ""


class KDLMTrainer():
    def __init__(self, args):
        self.model_name = "microsoft/deberta-base"
        self.stage = args.stage
        self.dataset_name = args.dataset
        self.pl_weight = args.pl_weight
        self.lr = args.lr
        self.seed = args.seed
        assert self.stage > 0
        self.prefix = "output" if self.stage > 1 else "prt_lm"
        self.prev_ckpt = f"{self.prefix}/{self.dataset_name}/bert.pt"
        self.ckpt = f"output/{self.dataset_name}/bert.pt"

    @ time_logger
    def train(self):
        # Preprocess data
        data, text = load_data(dataset=self.dataset_name, use_text=True)
        self.data = data
        self.num_nodes = data.x.shape[0]
        self.dim = feat_shrink if feat_shrink else 768
        self.n_labels = data.y.unique().size(0)
        pred_t = None
        emb_t = None
        if self.stage > 0:
            pred_t = np.memmap(f'output/{self.dataset_name}/gnn.pred',
                               mode='r',
                               dtype=np.float32,
                               shape=(self.num_nodes, self.n_labels))
            pred_t = torch.Tensor(np.array(pred_t))

            emb_t = np.memmap(f'output/{self.dataset_name}/gnn.emb',
                              mode='r',
                              dtype=np.float32,
                              shape=(self.num_nodes, self.dim))
            emb_t = torch.Tensor(np.array(emb_t))

        # Define pretrained tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        X = tokenizer(text, padding=True, truncation=True, max_length=512)
        self.dataset = KDDataset(
            X, data.y.tolist(), pred_t=pred_t, emb_t=emb_t)

        self.train_dataset = torch.utils.data.Subset(
            self.dataset, data.train_mask.nonzero().squeeze().tolist())
        self.val_dataset = torch.utils.data.Subset(
            self.dataset, data.val_mask.nonzero().squeeze().tolist())
        self.test_dataset = torch.utils.data.Subset(
            self.dataset, data.test_mask.nonzero().squeeze().tolist())

        bert_model = AutoModel.from_pretrained(self.model_name)
        bert_model.config.att_dropout = 0.1
        bert_model.config.dropout = 0.1
        self.model = KDBert(bert_model,
                            n_labels=self.n_labels,
                            is_augmented=self.stage > 0,
                            feat_shrink=feat_shrink,
                            dropout=0.4,
                            pseudo_label_weight=self.pl_weight)

        print(f"loading model from {self.prev_ckpt}")
        self.model.load_state_dict(torch.load(self.prev_ckpt))

        # Define Trainer
        log_steps = int(self.num_nodes/32*0.1)
        eval_steps = int(self.num_nodes/32*0.2)
        print("log_steps: ", log_steps)
        print("eval_steps: ", eval_steps)
        args = TrainingArguments(
            output_dir=f"output/{self.dataset_name}",
            do_train=True,
            do_eval=True,
            logging_steps=log_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_total_limit=1,
            eval_steps=eval_steps,
            save_steps=eval_steps,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8*8,
            num_train_epochs=1 if self.stage > 0 else 5,
            seed=self.seed,
            load_best_model_at_end=True,
            disable_tqdm=True,
            dataloader_num_workers=4,
            dataloader_drop_last=True,
            weight_decay=0.01,
            metric_for_best_model='loss',
            greater_is_better=False,
            # report_to="wandb"
            learning_rate=self.lr
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train pre-trained model
        self.trainer.train()
        torch.save(self.model.state_dict(), init_path(self.ckpt))

    @time_logger
    @torch.no_grad()
    def eval_and_save(self):
        emb = np.memmap(init_path(f"output/{self.dataset_name}/bert.emb"),
                        dtype=np.float32,
                        mode='w+',
                        shape=(self.num_nodes, self.dim))
        pred = np.memmap(init_path(f"output/{self.dataset_name}/bert.pred"),
                         dtype=np.float32,
                         mode='w+',
                         shape=(self.num_nodes, self.n_labels))
        inf_model = InfModel(self.model, emb, pred, feat_shrink=feat_shrink)
        inf_model.eval()
        inference_args = TrainingArguments(
            output_dir=f'output/',
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=64,
            dataloader_drop_last=False,
            dataloader_num_workers=4,
            fp16_full_eval=False,
            disable_tqdm=True,
        )

        trainer = Trainer(model=inf_model, args=inference_args)
        trainer.predict(self.dataset)
        if "ogbn" in self.dataset_name:
            from ogb.nodeproppred import Evaluator
            _evaluator = Evaluator(name=self.dataset_name)
        else:
            from core.GNNs.gnn_utils import Evaluator
            _evaluator = Evaluator(name=self.dataset_name)

        def evaluator(preds, labels): return _evaluator.eval({
            "y_true": torch.tensor(labels).view(-1, 1),
            "y_pred": torch.tensor(preds).view(-1, 1),
        })["acc"]

        def eval(x): return evaluator(
            np.argmax(pred[x], -1), self.data.y[x])

        res = {
            'lm_train_acc': eval(self.data.train_mask),
            'lm_val_acc': eval(self.data.val_mask),
            'lm_test_acc': eval(self.data.test_mask)}
        print(res)
