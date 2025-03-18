from sklearn.datasets import load_diabetes
from torch.utils.data import Dataset
import torch
import attrs

@attrs.define()
class DiabetesDataset(Dataset):
    """PyTorch Dataset wrapped around the sklearn diabetes dataset"""

    diabetes_dataset = load_diabetes()
    features = diabetes_dataset["data"]  # X
    labels = diabetes_dataset["target"]  # y
    feature_names = diabetes_dataset["feature_names"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        return torch.tensor(self.features[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)

