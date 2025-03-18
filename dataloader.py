from sklearn.datasets import load_diabetes
from torch.utils.data import Dataset
import torch
import attrs

@attrs.define()
class DiabetesDataset(Dataset):
    """PyTorch Dataset iterator for the sklearn diabetes dataset"""

    diabetes_dataset = load_diabetes()
    features = diabetes_dataset["data"]  # type: ignore
    labels = diabetes_dataset["target"]  # type: ignore
    feature_names = diabetes_dataset["feature_names"] # type: ignore

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.features[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)

