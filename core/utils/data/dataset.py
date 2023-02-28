import torch


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, features=None, gamma=None):
        self.encodings = encodings
        self.labels = labels
        self.features = features
        self.gamma = gamma

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['node_id'] = idx
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        if self.features is not None:
            item["features"] = self.features[idx]
        if self.gamma is not None:
            item["gamma"] = self.gamma[idx]
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class KDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, pred_t=None):
        self.encodings = encodings
        self.labels = labels
        self.pred_t = pred_t

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['node_id'] = idx
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        if self.pred_t is not None:
            item["pred_t"] = self.pred_t[idx]

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
