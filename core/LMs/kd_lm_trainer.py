import torch
import numpy as np
from core.utils.data.dataset import KDDataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, IntervalStrategy
from core.LMs.model import KDBert

from core.LMs.lm_utils import load_data
from core.LMs.lm_utils import compute_metrics
from core.utils.function.os_utils import init_path

feat_shrink = ""


class KDLMTrainer():
    def __init__(self, args):
        self.model_name = "microsoft/deberta-base"
        self.stage = args.stage
        self.dataset_name = args.dataset
        self.pl_weight = args.pl_weight

    def train(self):
        # Preprocess data
        data, text = load_data(dataset=self.dataset_name, use_text=True)
        self.num_nodes = data.x.shape[0]
        self.n_labels = data.y.unique().size(0)
        pred_t = None
        emb_t = None
        if self.stage > 0:
            pred_t = np.memmap(f'output/{self.dataset_name}/gnn.pred{self.stage-1}',
                               mode='r',
                               dtype=np.float32,
                               shape=(self.num_nodes, self.n_labels))
            pred_t = torch.Tensor(np.array(pred_t))

            emb_t = np.memmap(f'output/{self.dataset_name}/gnn.emb{self.stage-1}',
                              mode='r',
                              dtype=np.float32,
                              shape=(self.num_nodes, feat_shrink if feat_shrink else 768))
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

        self.model = KDBert(bert_model,
                            n_labels=self.n_labels,
                            is_augmented=self.stage > 0,
                            feat_shrink=feat_shrink,
                            pseudo_label_weight=self.pl_weight)

        if self.stage > 0:
            self.model.load_state_dict(torch.load(
                f"output/{self.dataset_name}/bert{self.stage-1}.pt"))

        # Define Trainer
        eval_steps = int(self.num_nodes/32*0.2)
        print("eval_steps: ", eval_steps)
        args = TrainingArguments(
            output_dir=f"output/{self.dataset_name}",
            do_train=True,
            do_eval=True,
            logging_steps=10,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_total_limit=3,
            eval_steps=eval_steps,
            save_steps=eval_steps,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8*8,
            num_train_epochs=5,
            seed=0,
            load_best_model_at_end=True,
            disable_tqdm=True,
            dataloader_num_workers=4,
            dataloader_drop_last=True,
            weight_decay=0.01,
            metric_for_best_model='accuracy',
            greater_is_better=True
            # learning_rate=2e-5
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

    @ torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model.state_dict(), init_path(
            f"output/{self.dataset_name}/bert{self.stage}.pt"))

        self.model.ckpt_emb = np.memmap(init_path(f"output/{self.dataset_name}/bert.emb{self.stage}"),
                                        dtype=np.float32,
                                        mode='w+',
                                        shape=(self.num_nodes, feat_shrink if feat_shrink else 768))
        self.model.ckpt_pred = np.memmap(init_path(f"output/{self.dataset_name}/bert.pred{self.stage}"),
                                         dtype=np.float32,
                                         mode='w+',
                                         shape=(self.num_nodes, self.n_labels))

        # Define Trainer
        args = TrainingArguments(
            output_dir="output",
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=8*8,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            disable_tqdm=True,
        )

        trainer = Trainer(model=self.model,
                          args=args,
                          compute_metrics=compute_metrics)

        train_metrics = trainer.predict(self.train_dataset).metrics
        val_metrics = trainer.predict(self.val_dataset).metrics
        test_metrics = trainer.predict(self.test_dataset).metrics

        train_metrics = {
            'train_'+k.split('_')[-1]: v for k, v in train_metrics.items()}
        val_metrics = {
            'val_'+k.split('_')[-1]: v for k, v in val_metrics.items()}

        print(train_metrics)
        print(val_metrics)
        print(test_metrics)

        # train_metrics = trainer.predict(self.dataset).metrics
        # print(train_metrics)
