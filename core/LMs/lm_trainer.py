import torch
from core.utils.data.dataset import Dataset
from transformers import BertTokenizer, AutoModel, TrainingArguments, Trainer, EarlyStoppingCallback
from core.LMs.model import BertEmb

from core.LMs.lm_utils import load_data
from core.LMs.lm_utils import compute_metrics


class LMTrainer():
    def __init__(self, ckpt):
        self.model_name = "bert-base-uncased"
        self.ckpt = ckpt

    def train(self):

        # Preprocess data
        data, text = load_data(dataset=self.dataset_name, use_text=True)

        # Define pretrained tokenizer and model
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        X = tokenizer(text, padding=True, truncation=True, max_length=512)
        dataset = Dataset(X, data.y.tolist())

        train_dataset = torch.utils.data.Subset(
            dataset, data.train_mask.nonzero().squeeze().tolist())
        val_dataset = torch.utils.data.Subset(
            dataset, data.val_mask.nonzero().squeeze().tolist())
        test_dataset = torch.utils.data.Subset(
            dataset, data.test_mask.nonzero().squeeze().tolist())

        bert_model = AutoModel.from_pretrained(self.model_name)
        model = BertEmb(bert_model, n_labels=data.y.unique().size(0), is_augmented=False)

        # Define Trainer
        args = TrainingArguments(
            output_dir="output",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=10,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            seed=0,
            load_best_model_at_end=True,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train pre-trained model
        trainer.train()
        torch.save(model.state_dict(), self.ckpt)
        metrics = trainer.predict(test_dataset).metrics
        print(metrics)
