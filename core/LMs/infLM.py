import torch
from core.utils.data.dataset import Dataset
from transformers import BertTokenizer, AutoModel, TrainingArguments, Trainer, EarlyStoppingCallback
from core.LMs.model import BertClassifier, BertClaInfModel
from core.data_utils.load_cora import get_raw_text_cora as get_raw_text
from core.LMs.lm_utils import compute_metrics

CKPT = "output/stage0.pt"


class LmInfTrainer():
    def __init__(self, ckpt):
        self.model_name = "bert-base-uncased"
        self.ckpt = ckpt

    def inference_pred_and_emb(self):
        data, text = get_raw_text(use_text=True)

        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        X = tokenizer(text, padding=True, truncation=True, max_length=512)
        dataset = Dataset(X, data.y.tolist())

        bert_model = AutoModel.from_pretrained(self.model_name)
        model = BertClassifier(bert_model, n_labels=data.y.unique().size(0))
        model.load_state_dict(torch.load(self.ckpt))
        inf_model = BertClaInfModel(model)

        inference_args = TrainingArguments(
            output_dir="output",
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=32,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            fp16_full_eval=True,
            disable_tqdm=False,
        )

        trainer = Trainer(
            model=inf_model,
            args=inference_args,
            compute_metrics=compute_metrics,
        )

        metrics = trainer.predict(dataset).metrics
        print(metrics)


if __name__ == "__main__":
    # ! Load data and train
    trainer = LmInfTrainer(ckpt=CKPT)
    trainer.inference_pred_and_emb()
