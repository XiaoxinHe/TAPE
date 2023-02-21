import torch


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, features=None):
        self.encodings = encodings
        self.labels = labels
        self.features = features

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['node_id'] = idx
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        if self.features is not None:
            item["features"] = self.features[idx]
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
