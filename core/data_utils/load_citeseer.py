import numpy as np
# adapted from https://github.com/jcatw/scnn
import torch
import random
import sklearn
import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import scipy.sparse as sp

# return citeseer dataset as pytorch geometric Data object together with 60/20/20 split, and list of citeseer IDs


def get_citeseer_casestudy():
    data_X, data_Y, data_citeid, data_edges = parse_citeseer()
    # data_X = sklearn.preprocessing.normalize(data_X, norm="l1")

    SEED = 0
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'citeseer'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('dataset', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)
    num_nodes_train = data.train_mask.sum()  # 120
    num_nodes_val = data.val_mask.sum()  # 150
    num_nodes_test = data.test_mask.sum()  # 1000
    data.train_id = np.sort(node_id[:num_nodes_train])
    data.val_id = np.sort(
        node_id[num_nodes_train: num_nodes_train+num_nodes_val])
    data.test_id = np.sort(node_id[-num_nodes_test:])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])

    return data, data_citeid

# credit: https://github.com/tkipf/pygcn/issues/27, xuhaiyun


def parse_citeseer():
    path = 'dataset/CiteSeer-Orig/citeseer'
    idx_features_labels = np.genfromtxt(
        "{}.content".format(path), dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(
        ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_citeseer(use_text=False):
    data, data_citeid = get_citeseer_casestudy()
    if not use_text:
        return data, None

    with open('dataset/CiteSeer-Orig/citeseer_texts.txt') as f:
        lines = f.read().splitlines()
    paper_ids = [lines[i] for i in range(len(lines)) if i % 3 == 0]
    abstracts = [lines[i] for i in range(len(lines)) if i % 3 == 1]
    # labels = [lines[i] for i in range(len(lines)) if i % 3 == 2]
    pid_ab = {}
    for i, j in zip(paper_ids, abstracts):
        pid_ab[i] = j
    text = []
    for pid in data_citeid:
        if pid in pid_ab:
            text.append(pid_ab[pid])
        else:
            text.append("None")
    return data, text
