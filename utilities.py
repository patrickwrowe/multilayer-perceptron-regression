from modules import Module
from dataloader import LinearReLUMLPDataSet
from trainer import Trainer
import numpy as np

def get_validation_predictions(model: Module, dataset: LinearReLUMLPDataSet) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the predictions of the model on the validation dataset.
    """
    y_hat = np.array([])
    y = np.array([])

    for batch in dataset.val_dataloader():
        y_hat = np.append(y_hat, model.net(batch[0]).detach().numpy())
        y = np.append(y, batch[-1])

    return y_hat, y

def extract_training_losses(metadata: dict) -> dict:
    """
    Extract the training and validation losses from the metadata dictionary. Of the training.Trainer class.

    e.g. metadata = {
        "max_epochs": 100,
        "num_train_batches": 10,
        "num_val_batches": 5,
        "training_epochs": [
            {
                "epoch": 0,
                "train_loss": [0.1, 0.2, 0.3, 0.4, 0.5],
                "val_loss": [0.2, 0.3, 0.4, 0.5, 0.6],
                "avg_train_loss": 0.3,
                "avg_val_loss": 0.4
            },
            ...
        ]
    }
    """

    train_losses = []
    avg_train_losses = []
    val_losses = []
    avg_val_losses = []

    for epoch in metadata["training_epochs"]:
        train_losses.append(epoch["train_loss"])
        avg_train_losses.append(epoch["avg_train_loss"])
        val_losses.append(epoch["val_loss"])
        avg_val_losses.append(epoch["avg_val_loss"])

    # return as dict of arrays
    return {
        "train_losses": np.array(train_losses),
        "avg_train_losses": np.array(avg_train_losses),
        "val_losses": np.array(val_losses),
        "avg_val_losses": np.array(avg_val_losses)
    }