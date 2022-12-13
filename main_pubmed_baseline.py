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

BATCH_SIZE = 32
N_SAMPLE = 19717


def train(model, data, optimizer):
    model.train()

    optimizer.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()
    out = model(data.x, data.edge_index)[data.train_mask]
    loss = criterion(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    out = model(data.x, data.edge_index)
    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y)

    train_acc = correct[data.train_mask].sum().item() / \
        data.train_mask.sum().item()
    val_acc = correct[data.val_mask].sum().item() / data.val_mask.sum().item()
    test_acc = correct[data.test_mask].sum().item() / \
        data.test_mask.sum().item()

    return train_acc, val_acc, test_acc, val_loss


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
    lr = 0.001
    gnn_model = GCN(
        in_channels=data.x.shape[1], hidden_channels=128, out_channels=3, num_layers=4, dropout=0)
    gnn_model.cuda()
    data.cuda()
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=lr,
                                                           patience=20,
                                                           verbose=True)

    best_test_perf = float('-inf')
    for epoch in range(1, 200):
        loss = train(gnn_model, data, optimizer)
        accs = test(gnn_model, data)
        scheduler.step(accs[-1])
        best_test_perf = accs[2] if accs[2] > best_test_perf else best_test_perf
        print(
            f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {accs[0]:.4f}, Val Acc: {accs[1]:.4f}, Test Acc: {best_test_perf:.4f}')


if __name__ == '__main__':
    main()
