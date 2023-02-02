import time
from transformers import BertTokenizer
import torch


def _preprocess(input_text, tokenizer):
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


def preprocessing(dataset, use_text=True):

    if dataset == 'cora':
        from core.data_utils.load_cora import get_raw_text_cora as get_raw_text
    elif dataset == 'pubmed':
        from core.data_utils.load_pubmed import get_raw_text_pubmed as get_raw_text
    elif dataset == 'citeseer':
        from core.data_utils.load_citeseer import get_raw_text_citeseer as get_raw_text
    elif dataset == 'ogbn-arxiv':
        from core.data_utils.load_arxiv import get_raw_text_arxiv as get_raw_text
    elif dataset == 'ogbn-products':
        from core.data_utils.load_products import get_raw_text_products as get_raw_text

    data, text = get_raw_text()
    if not use_text:
        return data

    print("[!] Preprocessing")
    start = time.time()
    token_id = []
    attention_masks = []
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case=True)
    for sample in text:
        encoding_dict = _preprocess(sample, tokenizer)
        token_id.append(encoding_dict['input_ids'])
        attention_masks.append(encoding_dict['attention_mask'])
    token_id = torch.cat(token_id, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    print(f"Time: {time.time()-start:.4f}")

    return data, token_id, attention_masks


def generate_node_embedding(model, dataloader, device):
    print("[!] Generating node embeddings")
    start = time.time()
    model.to(device)
    features = []
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        output = model(batch)
        features.append(output.detach().cpu())
    features = torch.cat(features, dim=0)
    print(f"Time: {time.time()-start:.4f}")
    return features
