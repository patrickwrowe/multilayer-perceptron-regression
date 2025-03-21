from pandas.compat import F
from scipy.sparse import data
from sklearn.datasets import load_diabetes
from torch.utils.data import Dataset, dataset
import torch
import torch.utils.data
import attrs
import pandas as pd
import os


@attrs.define()
class LinearReLUMLPDataSet(Dataset):
    """Abstract base class for datasets used in the LinearReLUMLP model"""

    batch_size: int

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_dataloader(self, train: bool):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)

    def get_data(self):
        raise NotImplementedError

@attrs.define()
class KFoldDataSet(Dataset):
    """Dataset for K-Fold Cross Validation. Takes the train_dataloader of the parent
    dataset and splits it into k folds"""

    dataset: LinearReLUMLPDataSet
    train_subset: torch.utils.data.Subset
    val_subset: torch.utils.data.Subset
    k: int
    n: int

    @classmethod
    def from_dataset(cls, dataset: LinearReLUMLPDataSet, k: int, n: int):
        def get_indices(k: int, n: int, train=True):      
            assert n <= k, "n must be less than or equal to k to fetch nth fold of a k-fold split."

            fold_size = len(dataset) // k
            all_indices = list(range(len(dataset)))
            
            if train:
                indices = all_indices[0 : n * fold_size] + all_indices[(n+1) * fold_size : -1]
            else: 
                indices = all_indices[n * fold_size : (n+1) * fold_size]
            
            return indices

        train_subset = torch.utils.data.Subset(dataset, indices=get_indices(k=k, n=n, train=True)) 
        val_subset = torch.utils.data.Subset(dataset, indices=get_indices(k=k, n=n, train=False))

        return cls(dataset=dataset, train_subset=train_subset, val_subset=val_subset, k=k, n=n)

    def get_dataloader(self, train):
        dataloader = self.train_subset if train else self.val_subset
        return dataloader


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
        return self.get_tensorloader((self.X, self.y), train, i)

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

    def get_dataloader(self, train):
        # Get the indices for the training and validation sets.
        i = (
            slice(0, int(len(self) - self.val_size))
            if train
            else slice(int(len(self) - self.val_size), None)
        )
        X = torch.tensor(self.features, dtype=torch.float32)
        y = torch.tensor(self.labels, dtype=torch.float32).reshape(-1, 1)
        return self.get_tensorloader((X, y), train, i)

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

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_dataloader(self, train: bool):
        label = "SalePrice"
        data = self.train if train else self.val

        # Sanity check here
        assert data is not None
        if label not in data:
            return

        get_tensor = lambda x: torch.tensor(x.values.astype(float), dtype=torch.float32)

        # Logarithm of prices
        tensors = (
            get_tensor(data.drop(columns=[label])),  # X
            torch.log(get_tensor(data[label])).reshape((-1, 1)), # Y
        )  
        return self.get_tensorloader(tensors, train)

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
