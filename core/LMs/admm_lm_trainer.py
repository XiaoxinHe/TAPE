import torch
import numpy as np
from core.utils.data.dataset import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, EarlyStoppingCallback
from core.LMs.model import ADMMBertEmb

from core.LMs.lm_utils import load_data
from core.LMs.lm_utils import compute_metrics
from core.utils.function.os_utils import init_path

feat_shrink = 128


class LMTrainer():
    def __init__(self, args):
        # self.model_name = "bert-base-uncased"
        self.model_name = "microsoft/deberta-base"
        self.stage = args.stage
        self.dataset_name = args.dataset
        self.penalty = 0.5

    def train(self):
        # Preprocess data
        data, text = load_data(dataset=self.dataset_name, use_text=True)
        self.num_nodes = data.x.shape[0]
        gamma = None

        if self.stage > 0:
            emb = np.memmap(f'output/{self.dataset_name}/z.emb{self.stage-1}', mode='r',
                            dtype=np.float32, shape=(self.num_nodes, feat_shrink))
            emb = torch.Tensor(np.array(emb))
            data.x = emb
            gamma = np.memmap(f'output/{self.dataset_name}/gamma.emb{self.stage-1}', mode='r',
                              dtype=np.float32, shape=(self.num_nodes, feat_shrink))

        # Define pretrained tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        X = tokenizer(text, padding=True, truncation=True, max_length=512)
        self.dataset = Dataset(
            X, data.y.tolist(), features=data.x, gamma=gamma)

        self.train_dataset = torch.utils.data.Subset(
            self.dataset, data.train_mask.nonzero().squeeze().tolist())
        self.val_dataset = torch.utils.data.Subset(
            self.dataset, data.val_mask.nonzero().squeeze().tolist())
        self.test_dataset = torch.utils.data.Subset(
            self.dataset, data.test_mask.nonzero().squeeze().tolist())

        bert_model = AutoModel.from_pretrained(self.model_name)
        bert_model.config.dropout = 0.1
        bert_model.config.attention_dropout = 0.1
        bert_model.config.cla_dropout = 0.1

        self.model = ADMMBertEmb(bert_model,
                                 n_labels=data.y.unique().size(0),
                                 is_augmented=self.stage > 0,
                                 feat_shrink=feat_shrink)

        if self.stage > 0:
            self.model.load_state_dict(torch.load(
                f"output/{self.dataset_name}/bert{self.stage-1}.pt"))

        print("self.model.is_augmented: ", self.model.is_augmented)

        # Define Trainer
        args = TrainingArguments(
            output_dir="output",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=1 if self.stage > 0 else 5,
            seed=0,
            load_best_model_at_end=True,
            disable_tqdm=True,
            dataloader_num_workers=1,
            dataloader_drop_last=True,
            weight_decay=0.01,
            learning_rate=2e-5
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset if self.stage > 0 else self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train pre-trained model
        self.trainer.train()

    @torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model.state_dict(), init_path(
            f"output/{self.dataset_name}/bert{self.stage}.pt"))
        ckpt_emb = np.memmap(init_path(f"output/{self.dataset_name}/bert.emb{self.stage}"), dtype=np.float32, mode='w+', shape=(
            self.num_nodes, feat_shrink if feat_shrink else 768))
        self.model.ckpt_emb = ckpt_emb

        # Define Trainer
        args = TrainingArguments(
            output_dir="output",
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=16,
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

        print(train_metrics)
        print(val_metrics)
        print(test_metrics)

        # train_metrics = trainer.predict(self.dataset).metrics
        # print(train_metrics)
