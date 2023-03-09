import argparse
import torch
import numpy as np
from core.utils.data.dataset import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, IntervalStrategy
from core.LMs.model import ADMMBert, InfModel

from core.LMs.lm_utils import load_data
from core.LMs.lm_utils import compute_metrics
from core.utils.function.os_utils import init_path

feat_shrink = ""


class AdmmInfLMTrainer():
    def __init__(self, args):
        self.model_name = "microsoft/deberta-base"
        self.stage = args.stage
        self.dataset_name = args.dataset
        self.seed = args.seed
        self.dim = feat_shrink if feat_shrink else 768
        self.path = args.path if args.path else f"output/{self.dataset_name}/bert{self.stage}.pt"

        if "ogbn" in self.dataset_name:
            from ogb.nodeproppred import Evaluator
            self._evaluator = Evaluator(name=self.dataset_name)
        else:
            from core.GNNs.gnn_utils import Evaluator
            self._evaluator = Evaluator(name=self.dataset_name)
        self.evaluator = lambda preds, labels: self._evaluator.eval({
            "y_true": torch.tensor(labels).view(-1, 1),
            "y_pred": torch.tensor(preds).view(-1, 1),
        })["acc"]

        data, text = load_data(dataset=self.dataset_name, use_text=True)
        self.data = data
        self.num_nodes = data.x.shape[0]
        self.n_labels = data.y.unique().size(0)

        self.emb = np.memmap(init_path(f"output/{self.dataset_name}/bert.emb{self.stage}"),
                             dtype=np.float32,
                             mode='w+',
                             shape=(self.num_nodes, self.dim))
        self.pred = np.memmap(init_path(f"output/{self.dataset_name}/bert.pred{self.stage}"),
                              dtype=np.float32,
                              mode='w+',
                              shape=(self.num_nodes, self.n_labels))

        # Define pretrained tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        X = tokenizer(text, padding=True, truncation=True, max_length=512)
        self.dataset = Dataset(X, data.y.tolist())

        self.train_dataset = torch.utils.data.Subset(
            self.dataset, data.train_mask.nonzero().squeeze().tolist())
        self.val_dataset = torch.utils.data.Subset(
            self.dataset, data.val_mask.nonzero().squeeze().tolist())
        self.test_dataset = torch.utils.data.Subset(
            self.dataset, data.test_mask.nonzero().squeeze().tolist())

        bert_model = AutoModel.from_pretrained(self.model_name)
        self.model = ADMMBert(bert_model,
                              n_labels=data.y.unique().size(0),
                              is_augmented=self.stage > 0,
                              feat_shrink=feat_shrink)
        
        print(f"loading model from {self.path}")
        self.model.load_state_dict(torch.load(self.path))

    @torch.no_grad()
    def inference_pred_and_emb(self):
        # torch.cuda.empty_cache()
        inf_model = InfModel(
            self.model, self.emb, self.pred, feat_shrink=feat_shrink)  # .to(self.cf.device)
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

        def eval(x): return self.evaluator(
            np.argmax(self.pred[x], -1), self.data.y[x])

        res = {
            'train_acc': eval(self.data.train_mask),
            'val_acc': eval(self.data.val_mask),
            'test_acc': eval(self.data.test_mask)}
        print(res)


if __name__ == "__main__":
    # ! Init Arguments
    parser = argparse.ArgumentParser(description='infLM')
    parser.add_argument('--path', type=str, default="")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="cora")

    args = parser.parse_args()
    print(f"\n\n[InfLM/{args.stage}]")
    trainer = AdmmInfLMTrainer(args)
    trainer.inference_pred_and_emb()
