import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F
import math
from bert import preprocessing, generate_node_embeddings

import json
from tqdm import tqdm
import pandas as pd
import time

from load_pubmed import get_pubmed_casestudy
from main_pubmed_gnn import GCN

BATCH_SIZE = 8
N_SAMPLE = 19717


def train(lm, gnn, g, loader, optimizer_lm, optimizer_gnn,  device):
    node_embs = []

    for batch in tqdm(loader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch
        output = lm(b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    output_hidden_states=True)
        emb = output['hidden_states'][-1]  # outputs[0]=last hidden state
        cls_token_emb = emb.permute(1, 0, 2)[0]
        node_embs.append(cls_token_emb.detach().cpu())
    node_embs = torch.cat(node_embs, dim=0)
    torch.cuda.empty_cache()  # PyTorch thing

    # train gnn

    X = node_embs.to(device)
    g = g.to(device)
    X.requires_grad = True
    X.retain_grad()
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(10):
        gnn.train()
        optimizer_gnn.zero_grad()
        out = gnn(X, g.edge_index)[g.train_mask]
        loss = criterion(out, g.y[g.train_mask])
        loss.backward()
        optimizer_gnn.step()
    torch.cuda.empty_cache()  # PyTorch thing

    grad = X.grad
    grad.requires_grad = True

    lm.train()
    optimizer_lm.zero_grad()
    for batch_idx, batch in enumerate(loader):
        start = time.time()
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch
        # Forward pass
        output = lm(b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    output_hidden_states=True)
        emb = output['hidden_states'][-1]  # outputs[0]=last hidden state
        cls_token_emb = emb.permute(1, 0, 2)[0]
        loss = grad[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE].sum()
        # loss.div_((math.ceil(N_SAMPLE/BATCH_SIZE)))
        loss.backward()
        print(
            f'Batch idx: {batch_idx:02d}, Loss: {loss.item()}, Time: {time.time()-start}')
        torch.cuda.empty_cache()  # PyTorch thing
    optimizer_lm.step()
    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()

    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y)

    train_acc = correct[data.train_mask].sum().item() / \
        data.train_mask.sum().item()
    val_acc = correct[data.val_mask].sum().item() / data.val_mask.sum().item()
    test_acc = correct[data.test_mask].sum().item() / \
        data.test_mask.sum().item()
    return train_acc, val_acc, test_acc


def main():
    # load data
    print("[!] Loading dataset")
    f = open('pubmed.json')
    pubmed = json.load(f)
    df_pubmed = pd.DataFrame.from_dict(pubmed)

    # Preprocess
    print("[!] Preprocessing")
    start = time.time()
    AB = df_pubmed['AB'].fillna("")
    TI = df_pubmed['TI'].fillna("")
    text = []
    for ti, ab in zip(TI, AB):
        t = 'Title: ' + ti + '\n'+'Abstract: ' + ab
        # t = ti + ab
        text.append(t)
    token_id = []
    attention_masks = []
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case=True)
    for sample in tqdm(text):
        encoding_dict = preprocessing(sample, tokenizer)
        token_id.append(encoding_dict['input_ids'])
        attention_masks.append(encoding_dict['attention_mask'])
    token_id = torch.cat(token_id, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    print("Time: ", time.time()-start)

    # Prepare DataLoader

    dataset = TensorDataset(token_id, attention_masks)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        sampler=SequentialSampler(dataset),
        batch_size=BATCH_SIZE
    )

    # Load the BertForSequenceClassification model
    bert = BertModel.from_pretrained(
        'bert-base-uncased',
        output_attentions=False,
        output_hidden_states=True,
    )

    # Run on GPU
    print("[!] Generating node embeddings")
    start = time.time()
    bert.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = generate_node_embeddings(bert, dataloader, device)
    print("Time: ", time.time()-start)

    data, data_pubid = get_pubmed_casestudy()
    data.x = features
    gnn_model = GCN(
        in_channels=data.x.shape[1], hidden_channels=128, out_channels=3, num_layers=4, dropout=0)
    gnn_model.cuda()

    print("[!] Start training")

    data.cuda()
    optimizer_gnn = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    optimizer_lm = torch.optim.Adam(bert.parameters(), lr=0.001)
    for epoch in range(1, 1000):
        start = time.time()
        loss = train(bert, gnn_model, data, dataloader,
                     optimizer_lm, optimizer_gnn, device)
        accs = test(gnn_model, data)
        torch.cuda.empty_cache()  # PyTorch thing
        print(
            f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {accs[0]:.4f}, Val Acc: {accs[1]:.4f}, Test Acc: {accs[2]:.4f}, Time: {time.time()-start}')


if __name__ == '__main__':
    main()
