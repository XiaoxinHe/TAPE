import time
from tqdm import tqdm
from transformers import BertTokenizer
from bert import generate_node_embeddings, preprocessing
import torch
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import pandas as pd

nodeidx2paperid = pd.read_csv(
    'dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')

raw_text = pd.read_csv('dataset/ogbn_arxiv/titleabs.tsv',
                       sep='\t', header=None, names=['paper id', 'title', 'abs'])
X = pd.merge(nodeidx2paperid, raw_text, on='paper id')


device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
dataset = PygNodePropPredDataset(
    name='ogbn-arxiv', transform=T.ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
data = data.to(device)

split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)

text = []
for ti, ab in zip(X['title'], X['abs']):
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

torch.save(token_id, 'token_id.pt')
torch.save(attention_masks, 'attention_masks.pt')