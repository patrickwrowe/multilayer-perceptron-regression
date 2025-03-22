# Multilayer Perceptron Regression

This repository contains a machine learning framework for exploring and training neural network models on various datasets. The project is built using PyTorch and provides utilities for data preprocessing, model training, and evaluation. It is designed to be modular and extensible, allowing for experimentation with different datasets and model architectures.

## Project Overview

The project is structured around three main components:

1. **Models**: Neural network architectures implemented as PyTorch modules.
2. **Dataloaders**: Custom dataset classes for loading and preprocessing data.
3. **Trainer**: A training loop implementation that handles model training, validation, and optimization.

Additionally, the project explores three datasets, each with its own unique characteristics and challenges:

1. **Diabetes Dataset**: A regression dataset from sklearn for predicting disease progression.
2. **Ames House Pricing Dataset**: A Kaggle dataset for predicting house prices.
3. **Synthetic Linear Dataset**: A synthetic dataset for debugging and testing linear regression models.

---

## Usage

### 1. **Trainer**

The `Trainer` class, defined in [`trainer.py`](trainer.py), is the core of the training process. It handles:

- Preparing the model, data, and training state.
- Running the training and validation loops.
- Logging metadata for analysis.

Example usage:

```python
from trainer import Trainer
from modules import LinearReLUMLP
from dataloader import DiabetesDataset

# Initialize dataset and model
data = SyntheticLinearData(weights=torch.tensor([0.25, 0.5]), batch_size=64)
model = LinearReLUMLP([32, 32], learning_rate=0.01)

# Initialize trainer and train the model
trainer = Trainer(max_epochs=10)
trainer.fit(model, data)
```

### 2. **Dataloaders**

The project defines several dataset classes in [`dataloader.py`](dataloader.py), all inheriting from the abstract base class `LinearReLUMLPDataSet`. These classes provide methods for loading, preprocessing, and batching data.

#### Key Dataset Classes:

- **`SyntheticLinearData`**: Generates synthetic data for linear regression, useful for debugging and testing.
- **`AmesHouse`**: Handles the Kaggle Ames House Pricing dataset, including preprocessing and feature engineering.
- **`DiabetesDataset`**: Loads the sklearn diabetes dataset. 

Each dataset class provides the following methods:
- `train_dataloader()`: Returns a PyTorch DataLoader for the training set.
- `val_dataloader()`: Returns a PyTorch DataLoader for the validation set.
- `get_tensors()`: Returns the raw feature and label tensors.

Example usage:

```python
from dataloader import DiabetesDataset

data = DiabetesDataset(batch_size=64)
train_loader = data.train_dataloader()
val_loader = data.val_dataloader()
```

#### K-Fold Cross Validation:

The `KFoldDataSet` class in [`dataloader.py`](dataloader.py) provides functionality for splitting datasets into training and validation folds. This is useful for evaluating model performance across multiple splits of the data.

Example usage:

```python
from dataloader import KFoldDataSet

data = KFoldDataSet(dataset, num_folds=5)
for fold in data:
    train_loader = fold.train_dataloader()
    val_loader = fold.val_dataloader()
```

### 3. **Models**

The project includes a modular neural network architecture, `LinearReLUMLP`, implemented in [`modules.py`](modules.py). This model consists of fully connected layers with ReLU activations and supports customizable architectures.

Example usage:

```python
from modules import LinearReLUMLP

model = LinearReLUMLP([32, 32], learning_rate=0.01, weight_decay=1e-5)
```

The above code creates a neural network with two hidden layers of size 32 and a learning rate of 0.01. Arbitrary layer configurations can be specified by providing a list of layer sizes, e.g., `[64, 32, 16]`. Regularization in the form of L2 weight decay can also be added by setting the `weight_decay` parameter.

---

## Datasets and Projects

### 1. **Diabetes Dataset**

- **Description**: A regression dataset from sklearn for predicting disease progression based on 10 features.
- **Implementation**: The dataset is loaded using the `DiabetesDataset` class in [`dataloader.py`](dataloader.py).
- **Exploration**: See [`diabetes_dataset/diabetes_dataset_eda.ipynb`](diabetes_dataset/diabetes_dataset_eda.ipynb) for exploratory data analysis and [`diabetes_dataset/diabetes_dataset_nn_models.ipynb`](diabetes_dataset/diabetes_dataset_nn_models.ipynb) for neural network experiments. It turns out that the dataset is not well-suited for neural networks due to its small size and low complexity, meaning that linear regression models perform as well or better than neural networks. A lesson learned is that neural networks are not always the best choice for every problem!

### 2. **Ames House Pricing Dataset**

- **Description**: A Kaggle dataset for predicting house prices based on various features.
- **Implementation**: The dataset is handled by the `AmesHouse` class in [`dataloader.py`](dataloader.py).
- **Exploration**: See [`ames_house_pricing_dataset/house_price_prediction.ipynb`](ames_house_pricing_dataset/house_price_prediction.ipynb) for experiments with this dataset.

### 3. **Synthetic Linear Dataset**

- **Description**: A synthetic dataset for linear regression, useful for debugging and testing.
- **Implementation**: The dataset is generated using the `SyntheticLinearData` class in [`dataloader.py`](dataloader.py).
- **Exploration**: See [`synthetic_linear_dataset/synthetic_linear_network.ipynb`](synthetic_linear_dataset/synthetic_linear_network.ipynb) for experiments with synthetic data.

---

## Key Features

- **K-Fold Cross Validation**: The `KFoldDataSet` class in [`dataloader.py`](dataloader.py) provides functionality for splitting datasets into training and validation folds.
- **Customizable Architectures**: The `LinearReLUMLP` model supports flexible layer configurations.
- **GPU Support**: The `Trainer` class automatically moves models to the GPU if available.