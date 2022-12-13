import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from transformers import PreTrainedModel
from main_pubmed_gnn import GCN
from main_pubmed_gnn import train as train_gnn, test as test_gnn


def preprocessing(input_text, tokenizer):
    '''
    Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
    '''
    return tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=128,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )


def generate_node_embeddings(model, loader, device):
    node_embs = []
    for batch in tqdm(loader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch
        # Forward pass
        output = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask,
                       output_hidden_states=True)
        emb = output['hidden_states'][-1]  # outputs[0]=last hidden state
        cls_token_emb = emb.permute(1, 0, 2)[0]
        node_embs.append(cls_token_emb.detach().cpu())
    return torch.cat(node_embs, dim=0)


class Bert(PreTrainedModel):
    def __init__(self, model):
        super().__init__()
        self.bert_encoder = model

    def forward(self, batch):
        b_input_ids, b_input_mask = batch
        output = self.bert_encoder(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   output_hidden_states=True)
        emb = output['hidden_states'][-1]  # outputs[0]=last hidden state
        cls_token_emb = emb.permute(1, 0, 2)[0]
        return cls_token_emb


class LM_GNN(nn.Module):
    def __init__(self):
        super().__init__()
        model = BertModel.from_pretrained(
            'bert-base-uncased',
            output_attentions=False,
            output_hidden_states=True)
        self.lm = Bert(model)
        self.gnn = GCN(in_channels=768, hidden_channels=128,
                       out_channels=3, num_layers=4, dropout=0)

    def forward(self, data):
        self.lm()
        self.gnn(data)
