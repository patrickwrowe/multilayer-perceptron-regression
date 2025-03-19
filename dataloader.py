from sklearn.datasets import load_diabetes
from torch.utils.data import Dataset
import torch
from torch import tensor, utils
import torch.utils.data
import attrs
import pandas as pd


import requests
import os
import hashlib
import zipfile
import tarfile


@attrs.define()
class DiabetesDataset(Dataset):
    """PyTorch Dataset iterator for the sklearn diabetes dataset"""

    batch_size: int = 64
    val_size: float = 0.2
    manual_seed: int = 42
    diabetes_dataset = load_diabetes()

    # Only load relevant features
    features = diabetes_dataset["data"][:, [0, 2, 3, 4, 5, 6, 7, 8, 9]]  # type: ignore
    labels = diabetes_dataset["target"]  # type: ignore
    feature_names = diabetes_dataset["feature_names"] # type: ignore
    
    def __attrs_post_init__(self):
        self.features = self.features - self.features.mean(axis=0) / self.features.std(axis=0)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor(self.features[idx], dtype=torch.float), 
                torch.tensor(self.labels[idx], dtype=torch.float))

    def get_dataloader(self, train):
        i = slice(0, 350) if train else slice(350, None)
        X = torch.tensor(self.features, dtype=torch.float32)
        y = torch.tensor(self.labels, dtype=torch.float32).reshape(-1, 1)
        return self.get_tensorloader((X, y), train, i)

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)
    

class SyntheticLinearData():
    """
    
    Synthetic data for linear regression.
    
    Useful for debugging
    """

    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32):
        self.num_train = num_train
        self.num_val = num_val
        self.batch_size = batch_size
        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, torch.reshape(w, (-1, 1))) + b + noise

    def get_dataloader(self, train):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)




#### House prices data


def download(url, folder='../data', sha1_hash=None):
    """Download a file to folder and return the local filepath.

    Defined in :numref:`sec_utils`"""
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('/')[-1])
    # Check if hit cache
    if os.path.exists(fname) and sha1_hash:
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    # Download
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def extract(filename, folder=None):
    """Extract a zip/tar file into folder.

    Defined in :numref:`sec_utils`"""
    base_dir = os.path.dirname(filename)
    _, ext = os.path.splitext(filename)
    assert ext in ('.zip', '.tar', '.gz'), 'Only support zip/tar files.'
    if ext == '.zip':
        fp = zipfile.ZipFile(filename, 'r')
    else:
        fp = tarfile.open(filename, 'r')
    if folder is None:
        folder = base_dir
    fp.extractall(folder)

class KaggleHouse():

    def __init__(self, batch_size, train=None, val=None):
        super().__init__()

        self.batch_size = batch_size
        self.root = '../data'
        self.train = train
        self.val = val

        if self.train is None:
            self.raw_train = pd.read_csv(download(
                "http://d2l-data.s3-accelerate.amazonaws.com/" + "kaggle_house_pred_train.csv", self.root,
                sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            self.raw_val = pd.read_csv(download(
                "http://d2l-data.s3-accelerate.amazonaws.com/"  + 'kaggle_house_pred_test.csv', self.root, 
                sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))


    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)
    
    def get_dataloader(self, train):
        label = 'SalePrice'
        data = self.train if train else self.val
        if label not in data: return
        get_tensor = lambda x: torch.tensor(x.values.astype(float),
        dtype=torch.float32)
        # Logarithm of prices
        tensors = (get_tensor(data.drop(columns=[label])), # X
        torch.log(get_tensor(data[label])).reshape((-1, 1))) # Y
        return self.get_tensorloader(tensors, train)


    def preprocess(self):
        #Remove the ID and label columns
        label = 'SalePrice'
        features = pd.concat(
            (self.raw_train.drop(columns=['Id', label]),
            self.raw_val.drop(columns=['Id'])))

        # Standardize numerical columns
        numeric_features = features.dtypes[features.dtypes != 'object'].index
        features[numeric_features] = features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))

        # Replace NaN numerical features by 0.
        features[numeric_features] = features[numeric_features].fillna(0)

        # Get one-hot encoding
        features = pd.get_dummies(features, dummy_na=True)

        # Split back out knowing we concatentated train/val to perform preprocessing concisely. 
        self.train = features[:self.raw_train.shape[0]].copy()
        self.train[label] = self.raw_train[label]
        # Validation starts when train ends. 
        self.val = features[self.raw_train.shape[0]:].copy()

