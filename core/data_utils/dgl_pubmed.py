import torch
import random
import dgl
from dgl.data import DGLDataset
import numpy as np


class PubMedDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="pubmed")

    def _parse(self):
        path = 'dataset/Pubmed-Diabetes/data/'

        n_nodes = 19717
        n_features = 500
        n_classes = 3

        data_X = np.zeros((n_nodes, n_features), dtype='float32')
        data_Y = [None] * n_nodes
        data_pubid = [None] * n_nodes
        data_edges = []

        paper_to_index = {}
        feature_to_index = {}

        # parse nodes
        with open(path + 'Pubmed-Diabetes.NODE.paper.tab', 'r') as node_file:
            # first two lines are headers
            node_file.readline()
            node_file.readline()

            k = 0

            for i, line in enumerate(node_file.readlines()):
                items = line.strip().split('\t')

                paper_id = items[0]
                data_pubid[i] = paper_id
                paper_to_index[paper_id] = i

                # label=[1,2,3]
                label = int(items[1].split('=')[-1]) - \
                    1  # subtract 1 to zero-count
                data_Y[i] = label

                # f1=val1 \t f2=val2 \t ... \t fn=valn summary=...
                features = items[2:-1]
                for feature in features:
                    parts = feature.split('=')
                    fname = parts[0]
                    fvalue = float(parts[1])

                    if fname not in feature_to_index:
                        feature_to_index[fname] = k
                        k += 1

                    data_X[i, feature_to_index[fname]] = fvalue

        # parse graph
        data_A = np.zeros((n_nodes, n_nodes), dtype='float32')

        with open(path + 'Pubmed-Diabetes.DIRECTED.cites.tab', 'r') as edge_file:
            # first two lines are headers
            edge_file.readline()
            edge_file.readline()

            for i, line in enumerate(edge_file.readlines()):

                # edge_id \t paper:tail \t | \t paper:head
                items = line.strip().split('\t')

                edge_id = items[0]

                tail = items[1].split(':')[-1]
                head = items[3].split(':')[-1]

                data_A[paper_to_index[tail], paper_to_index[head]] = 1.0
                data_A[paper_to_index[head], paper_to_index[tail]] = 1.0
                if head != tail:
                    data_edges.append(
                        (paper_to_index[head], paper_to_index[tail]))
                    data_edges.append(
                        (paper_to_index[tail], paper_to_index[head]))

        return data_X, data_Y, data_pubid, np.unique(data_edges, axis=0).transpose()

    def process(self):
        nodes_data, node_labels, data_citeid, edges_data = self._parse()
        edges_src, edges_dst = torch.from_numpy(edges_data)

        self.graph = dgl.graph(
            (edges_src, edges_dst), num_nodes=nodes_data.shape[0]
        )
        self.graph.ndata["feat"] = torch.from_numpy(nodes_data)
        self.graph.ndata["label"] = torch.from_numpy(np.array(node_labels))
        self.graph.citeid = data_citeid
        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.

        SEED = 0
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)  # Numpy module.
        random.seed(SEED)  # Python random module.

        # split data
        num_nodes = nodes_data.shape[0]
        node_id = np.arange(num_nodes)
        np.random.shuffle(node_id)

        train_id = np.sort(node_id[:int(num_nodes * 0.6)])
        val_id = np.sort(node_id[int(num_nodes * 0.6):int(num_nodes * 0.8)])
        test_id = np.sort(node_id[int(num_nodes * 0.8):])

        self.idx_split = {}
        self.idx_split["train"] = torch.from_numpy(train_id)
        self.idx_split["valid"] = torch.from_numpy(val_id)
        self.idx_split["test"] = torch.from_numpy(test_id)

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

    def get_idx_split(self):
        return self.idx_split
