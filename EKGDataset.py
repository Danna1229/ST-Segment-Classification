import numpy as np
from torch.utils.data import Dataset
import torch

class EKGDataset(Dataset):
    def __init__(self, data_path, label_1_path,label_12_path):
        super(EKGDataset, self).__init__()

        # Load data and labels
        self.data = np.load(data_path)
        self.label_1 = np.load(label_1_path)
        self.label_12 = np.load(label_12_path)

    def __len__(self):
        return len(self.label_1)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y_1 = torch.tensor(self.label_1[index], dtype=torch.long)
        y_12 = torch.tensor(self.label_12[index], dtype=torch.long)
        return x, y_1,y_12