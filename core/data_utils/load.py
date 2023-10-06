import os
import json
import torch
import csv
from core.data_utils.dataset import CustomDGLDataset


def load_gpt_preds(dataset, topk):
    preds = []
    fn = f'gpt_preds/{dataset}.csv'
    print(f"Loading topk preds from {fn}")
    with open(fn, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            inner_list = []
            for value in row:
                inner_list.append(int(value))
            preds.append(inner_list)

    pl = torch.zeros(len(preds), topk, dtype=torch.long)
    for i, pred in enumerate(preds):
        pl[i][:len(pred)] = torch.tensor(pred[:topk], dtype=torch.long)+1
    return pl


def load_data(dataset, use_dgl=False, use_text=False, use_gpt=False, seed=0):
    if dataset == 'cora':
        from core.data_utils.load_cora import get_raw_text_cora as get_raw_text
        num_classes = 7
    elif dataset == 'pubmed':
        from core.data_utils.load_pubmed import get_raw_text_pubmed as get_raw_text
        num_classes = 3
    elif dataset == 'ogbn-arxiv':
        from core.data_utils.load_arxiv import get_raw_text_arxiv as get_raw_text
        num_classes = 40
    elif dataset == 'ogbn-products':
        from core.data_utils.load_products import get_raw_text_products as get_raw_text
        num_classes = 47
    elif dataset == 'arxiv_2023':
        from core.data_utils.load_arxiv_2023 import get_raw_text_arxiv_2023 as get_raw_text
        num_classes = 40
    else:
        exit(f'Error: Dataset {dataset} not supported')

    # for training GNN
    if not use_text:
        data, _ = get_raw_text(use_text=False, seed=seed)
        if use_dgl:
            data = CustomDGLDataset(dataset, data)
        return data, num_classes

    # for finetuning LM
    if use_gpt:
        data, text = get_raw_text(use_text=False, seed=seed)
        folder_path = 'gpt_responses/{}'.format(dataset)
        print(f"using gpt: {folder_path}")
        n = data.y.shape[0]
        text = []
        for i in range(n):
            filename = str(i) + '.json'
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                content = json_data['choices'][0]['message']['content']
                text.append(content)
    # if use_gpt:
    #     data, text = get_raw_text(use_text=False, seed=seed)
    #     folder_path = 'llama_responses/{}'.format(dataset)
    #     print(f"using gpt: {folder_path}")
    #     n = data.y.shape[0]
    #     text = []
    #     for i in range(n):
    #         filename = str(i) + '.json'
    #         file_path = os.path.join(folder_path, filename)
    #         with open(file_path, 'r') as file:
    #             json_data = json.load(file)
    #             content = json_data['generation']['content']
    #             text.append(content)
    else:
        data, text = get_raw_text(use_text=True, seed=seed)

    return data, num_classes, text
