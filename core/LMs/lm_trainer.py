import torch
import numpy as np
from core.utils.data.dataset import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
from core.LMs.model import BertClassifier, BertClaInfModel

from core.LMs.lm_utils import load_data
from core.LMs.lm_utils import compute_metrics
from core.utils.function.os_utils import init_path, time_logger


class LMTrainer():
    def __init__(self, args):
        self.model_name = args.model
        self.dataset_name = args.dataset
        self.seed = args.seed

        self.feat_shrink = args.feat_shrink
        self.weight_decay = args.weight_decay
        self.dropout = args.dropout
        self.att_dropout = args.att_dropout
        self.cla_dropout = args.cla_dropout

        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.warmup_epochs = args.warmup_epochs
        self.eval_patience = args.eval_patience
        self.lr = args.lr

        self.ckpt = f"output/{self.dataset_name}/{self.model_name}.ckpt"

        # Preprocess data
        data, text = load_data(dataset=self.dataset_name, use_text=True)
        self.num_nodes = data.x.size(0)
        self.n_labels = data.y.unique().size(0)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        X = tokenizer(text, padding=True, truncation=True, max_length=512)
        dataset = Dataset(X, data.y.tolist())

        self.train_dataset = torch.utils.data.Subset(
            dataset, data.train_mask.nonzero().squeeze().tolist())
        self.val_dataset = torch.utils.data.Subset(
            dataset, data.val_mask.nonzero().squeeze().tolist())
        self.test_dataset = torch.utils.data.Subset(
            dataset, data.test_mask.nonzero().squeeze().tolist())

        self.data = data
        self.inf_dataset = dataset

    def train(self):
        train_steps = self.num_nodes // self.batch_size + 1
        eval_steps = self.eval_patience // self.batch_size
        warmup_steps = int(self.warmup_epochs * train_steps)

        # Define pretrained tokenizer and model
        bert_model = AutoModel.from_pretrained(self.model_name)
        self.model = BertClassifier(bert_model,
                                    n_labels=self.n_labels,
                                    feat_shrink=self.feat_shrink)
        self.model.config.dropout = self.dropout
        self.model.config.attention_dropout = self.att_dropout

        # Define Trainer
        args = TrainingArguments(
            output_dir=f'output/{self.dataset_name}/{self.model_name}',
            do_train=True,
            do_eval=True,
            eval_steps=eval_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_steps=eval_steps,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            save_total_limit=1,
            load_best_model_at_end=True,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size*8,
            warmup_steps=warmup_steps,
            num_train_epochs=self.epochs,
            dataloader_num_workers=1,
            fp16=True,
            dataloader_drop_last=True,
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train pre-trained model
        self.trainer.train()
        torch.save(self.model.state_dict(), init_path(self.ckpt))

    @time_logger
    @torch.no_grad()
    def eval_and_save(self):
        emb = np.memmap(init_path(f"output/{self.dataset_name}/{self.model_name}.emb"),
                        dtype=np.float16,
                        mode='w+',
                        shape=(self.num_nodes, self.feat_shrink if self.feat_shrink else 768))
        pred = np.memmap(init_path(f"output/{self.dataset_name}/{self.model_name}.pred"),
                         dtype=np.float16,
                         mode='w+',
                         shape=(self.num_nodes, self.n_labels))

        inf_model = BertClaInfModel(
            self.model, emb, pred, feat_shrink=self.feat_shrink)
        inf_model.eval()
        inference_args = TrainingArguments(
            output_dir=f'output/{self.dataset_name}/{self.model_name}',
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.batch_size*8,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            fp16_full_eval=True,
        )

        trainer = Trainer(model=inf_model, args=inference_args)
        trainer.predict(self.inf_dataset)
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
