import os
from ogb.nodeproppred import PygNodePropPredDataset
import gdown
from ogb.utils.url import extract_zip
import pandas as pd
import torch_geometric.transforms as T


def get_raw_text_products():

    dataset = PygNodePropPredDataset(
        'ogbn-products', transform=T.ToSparseTensor())
    labels = dataset[0].y

    
    raw_text_url = 'https://drive.google.com/u/0/uc?id=1gsabsx8KR2N9jJz16jTcA0QASXsNuKnN&export=download'
    data_root = '/home/xiaoxin/TAG/dataset/ogbn_products'

    opath = os.path.join(raw_text_url, data_root)
    print(os.path.join(opath, "Amazon-3M.raw"))
    output = os.path.join(opath, 'Amazon-3M.raw.zip')
    if not os.path.exists(os.path.join(opath, "Amazon-3M.raw.zip")):
        url = raw_text_url
        gdown.download(url=url, output=output, quiet=False, fuzzy=False)
    if not os.path.exists(os.path.join(opath, "Amazon-3M.raw")):
        extract_zip(output, opath)
    raw_text_path = os.path.join(opath, "Amazon-3M.raw")

    def read_mappings(data_root):
        category_path_csv = f"{data_root}/mapping/labelidx2productcategory.csv.gz"
        products_asin_path_csv = f"{data_root}/mapping/nodeidx2asin.csv.gz"  #
        products_ids = pd.read_csv(products_asin_path_csv)
        categories = pd.read_csv(category_path_csv)
        # categories.columns = ["ID", "category"]  # 指定ID 和 category列写进去
        return categories, products_ids

    def get_mapping_product(labels, meta_data: pd.DataFrame, products_ids: pd.DataFrame, categories):
        # ! Read mappings for OGBN-products
        products_ids.columns = ["ID", "asin"]
        categories.columns = ["label_idx", "category"]
        meta_data.columns = ['asin', 'title', 'content']
        products_ids["label_idx"] = labels
        data = pd.merge(products_ids, meta_data, how="left",
                        on="asin")  # ID ASIN TITLE
        data = pd.merge(data, categories, how="left", on="label_idx")
        # ID ASIN LABEL_IDX TITLE CATEGORY
        return data

    def read_product_json(raw_text_path):
        import json
        import gzip
        if not os.path.exists(os.path.join(raw_text_path, "trn.json")):
            trn_json = os.path.join(raw_text_path, "trn.json.gz")
            trn_json = gzip.GzipFile(trn_json)
            open(os.path.join(raw_text_path, "trn.json"),
                 "wb+").write(trn_json.read())
            os.unlink(os.path.join(raw_text_path, "trn.json.gz"))
            tst_json = os.path.join(raw_text_path, "tst.json.gz")
            tst_json = gzip.GzipFile(tst_json)
            open(os.path.join(raw_text_path, "tst.json"),
                 "wb+").write(tst_json.read())
            os.unlink(os.path.join(raw_text_path, "tst.json.gz"))

        i = 1
        for root, _, files in os.walk(os.path.join(raw_text_path, '')):
            for file in files:
                if not '.json' in file:
                    continue
                file_path = os.path.join(root, file)
                print(file_path)
                with open(file_path, 'r', encoding='utf_8_sig') as file_in:
                    title = []
                    for line in file_in.readlines():
                        dic = json.loads(line)
                        dic['title'] = dic['title'].strip("\"\n")
                        title.append(dic)
                    name_attribute = ["uid", "title", "content"]
                    writercsv = pd.DataFrame(
                        columns=name_attribute, data=title)
                    writercsv.to_csv(os.path.join(
                        raw_text_path, f'product' + str(i) + '.csv'), index=False, encoding='utf_8_sig')
                    i = i + 1
        return

    def read_meta_product(raw_text_path):
        if not os.path.exists(os.path.join(raw_text_path, f'product3.csv')):
            read_product_json(raw_text_path)
            path_product1 = os.path.join(raw_text_path, f'product1.csv')
            path_product2 = os.path.join(raw_text_path, f'product2.csv')
            pro1 = pd.read_csv(path_product1)
            pro2 = pd.read_csv(path_product2)
            file = pd.concat([pro1, pro2])
            file.drop_duplicates()
            file.to_csv(os.path.join(raw_text_path,
                        f'product3.csv'), index=False, sep=" ")
        else:
            file = pd.read_csv(os.path.join(
                raw_text_path, 'product3.csv'), sep=" ")

        return file

    print('Loading raw text')
    meta_data = read_meta_product(raw_text_path)  # 二维表
    categories, products_ids = read_mappings(data_root)
    node_data = get_mapping_product(
        labels, meta_data, products_ids, categories)  # 返回拼接后的数据

    del meta_data, categories, products_ids
    text_func = {
        'T': lambda x: x['title'],
        'TC': lambda x: f"Title: {x['title']}. Content: {x['content']}",
    }

    process_mode = 'TC'
    cut_off = 256
    node_data['text'] = node_data.apply(text_func[process_mode], axis=1)
    node_data['text'] = node_data.apply(lambda x: ' '.join(
        str(x['text']).split(' ')[:cut_off]), axis=1)
    node_data = node_data[['ID', 'text']]

    return dataset[0], node_data['text']
