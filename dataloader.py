from sklearn.datasets import load_diabetes
from torch.utils.data import Dataset
import torch
from torch import tensor, utils
import torch.utils.data
import attrs

@attrs.define()
class DiabetesDataset(Dataset):
    """PyTorch Dataset iterator for the sklearn diabetes dataset"""

    batch_size: int = 64
    diabetes_dataset = load_diabetes()
    features = diabetes_dataset["data"]  # type: ignore
    labels = diabetes_dataset["target"]  # type: ignore
    feature_names = diabetes_dataset["feature_names"] # type: ignore

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor(self.features[idx], dtype=torch.float), 
                torch.tensor(self.labels[idx], dtype=torch.float))

    def get_dataloader(self, train):
        get_tensor = lambda x: torch.tensor(x.values.astype(float),
                                            dtype=torch.float32)

        tensors = (get_tensor(self.features), get_tensor(self.labels))
        return self.get_tensorloader(tensors, train)

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)