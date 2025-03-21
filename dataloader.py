
from sklearn.datasets import load_diabetes
from torch.utils.data import Dataset
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, Subset
import attrs
import pandas as pd
import os
from typing import Union


@attrs.define()
class LinearReLUMLPDataSet(Dataset):
    """Abstract base class for datasets used in the LinearReLUMLP model"""

    batch_size: int

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_tensors(self):
        raise NotImplementedError

    def get_dataloader(self, train: bool):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, train, indices: Union[list[int], slice]=slice(0, None)):
        tensors = tuple(a[indices] for a in self.get_tensors())
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)

    def get_data(self):
        raise NotImplementedError

@attrs.define()
class KFoldDataSet(LinearReLUMLPDataSet):
    """Dataset for K-Fold Cross Validation. Takes the train_dataloader of the parent
    dataset and splits it into k folds
    
    After writing this I learned about PyTorch Samplers, which is probably a _much_
    better way to do implement this: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler
    
    But hey-ho, this is just a learning project, so we'll use my hacky implementation."""

    train_subset_dataloader: torch.utils.data.DataLoader
    val_subset_dataloader: torch.utils.data.DataLoader
    k: int
    n: int

    @classmethod
    def from_dataset(cls, dataset: LinearReLUMLPDataSet, k: int, n: int):
        def get_indices(length: int, k: int, n: int, train=True):      
            assert n <= k, "n must be less than or equal to k to fetch nth fold of a k-fold split."

            fold_size = length // k
            all_indices = list(range(length))
            
            indices = all_indices[0 : n * fold_size] + all_indices[(n+1) * fold_size : -1] if train else all_indices[n * fold_size : (n+1) * fold_size]
            return indices

        # This is a horrible way to get the total length of the training dataloader. So ineffient.
        train_length = sum(len(batch[0]) for batch in dataset.train_dataloader())
        train_subset = dataset.get_tensorloader(
            train=True,
            indices=get_indices(train_length, k=k, n=n, train=True)
        )
        val_subset = dataset.get_tensorloader(
            train=False, 
            indices=get_indices(train_length, k=k, n=n, train=False)
        )

        return cls(batch_size=dataset.batch_size, train_subset_dataloader=train_subset, val_subset_dataloader=val_subset, k=k, n=n)

    def get_dataloader(self, train):
        return self.train_subset_dataloader if train else self.val_subset_dataloader


@attrs.define()
class SyntheticLinearData(LinearReLUMLPDataSet):
    """

    Synthetic data for linear regression.

    Useful for debugging
    """

    weights: torch.Tensor
    bias: float
    noise_scale: float = 0.01
    num_train: int = 1000
    num_val: int = 1000
    batch_size: int = 32

    def __attrs_post_init__(self):
        n = self.num_train + self.num_val
        self.X = torch.randn(n, len(self.weights))
        noise = torch.randn(n, 1) * self.noise_scale
        self.y = (
            torch.matmul(self.X, torch.reshape(self.weights, (-1, 1)))
            + self.bias
            + noise
        )

    def get_dataloader(self, train: bool):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader(train, i)

    def get_tensors(self):
        return self.X, self.y

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)


@attrs.define()
class DiabetesDataset(LinearReLUMLPDataSet):
    """PyTorch Dataset iterator for the sklearn diabetes dataset"""

    batch_size: int = 64
    val_size: float = 0.2
    manual_seed: int = 42
    diabetes_dataset = load_diabetes()

    features = diabetes_dataset["data"]  # type: ignore
    labels = diabetes_dataset["target"]  # type: ignore
    feature_names = diabetes_dataset["feature_names"]  # type: ignore

    def __attrs_post_init__(self):
        self.features = self.features - self.features.mean(axis=0) / self.features.std(
            axis=0
        )
        self.val_size_abs = int(len(self) * self.val_size)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.features[idx], dtype=torch.float),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )

    def get_tensors(self):
        X = torch.tensor(self.features, dtype=torch.float32)
        y = torch.tensor(self.labels, dtype=torch.float32).reshape(-1, 1)
        return X, y

    def get_dataloader(self, train):
        # Get the indices for the training and validation sets.
        i = (
            slice(0, int(len(self) - self.val_size_abs))
            if train
            else slice(int(len(self) - self.val_size_abs), None)
        )
        return self.get_tensorloader(train, i)

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)


@attrs.define()
class KaggleHouse(LinearReLUMLPDataSet):

    batch_size: int = 32
    root = "../data"
    train = None
    val = None

    def __attrs_post_init__(self):
        self.raw_train = pd.read_csv(os.path.join(self.root, "kaggle_house_train.csv"))
        self.raw_val = pd.read_csv(os.path.join(self.root, "kaggle_house_test.csv"))

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_dataloader(self, train: bool):
        if train: 
            return self.get_tensorloader(train)
        else:
            return

    def get_tensors(self):
        label = "SalePrice"
        get_tensor = lambda x: torch.tensor(x.values.astype(float), dtype=torch.float32)
        assert self.train is not None
        X = get_tensor(self.train.drop(columns=[label]))  # X
        y = torch.log(get_tensor(self.train[label])).reshape((-1, 1)) # Y
        return X, y

    def preprocess(self):
        # Remove the ID and label columns
        label = "SalePrice"
        features = pd.concat(
            (
                self.raw_train.drop(columns=["Id", label]),
                self.raw_val.drop(columns=["Id"]),
            )
        )

        # Standardize numerical columns
        numeric_features = features.dtypes[features.dtypes != "object"].index
        features[numeric_features] = features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std())
        )

        # Replace NaN numerical features by 0.
        features[numeric_features] = features[numeric_features].fillna(0)

        # Get one-hot encoding
        features = pd.get_dummies(features, dummy_na=True)

        self.train = features[: self.raw_train.shape[0]].copy()
        self.train[label] = self.raw_train[label]

        # Validation starts when train ends.
        self.val = features[self.raw_train.shape[0] :].copy()
