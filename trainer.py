import attrs
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim.optimizer import Optimizer
import numpy as np
from typing import Optional

@attrs.define(slots=False)
class Trainer:

    # Training Options
    max_epochs: int
    init_random: Optional[int] = None

    # Model paramaters
    model: Module = attrs.field(init=False)
    optim: Optimizer = attrs.field(init=False)

    # Data Parameters
    train_dataloader: DataLoader = attrs.field(init=False)
    val_dataloader: DataLoader = attrs.field(init=False)
    num_train_batches: int = attrs.field(init=False)
    num_val_batches: int = attrs.field(init=False)

    # Counters
    epoch: int = attrs.field(init=False)
    train_batch_idx: int = attrs.field(init=False)
    val_batch_idx: int = attrs.field(init=False)

    # Metadata
    metadata: dict = {}

    def prepare_state(self):
        if self.init_random:
            torch.manual_seed(self.init_random)
            torch.use_deterministic_algorithms(True)

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (
            len(self.val_dataloader) if self.val_dataloader is not None else 0
        )

    def prepare_model(self, model):
        model.trainer = self
        # Attempt to put the model on the GPU if one is available
        device = try_gpu()
        model.to(device)
        self.model = model
        print(f"Model running on {device}")

    def fit(self, model, data):
        self.prepare_state()
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0

        self.metadata = {
            "max_epochs": self.max_epochs,
            "num_train_batches": self.num_train_batches,
            "num_val_batches": self.num_val_batches,
            "training_epochs": [],
        }
        
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        """Fit one epoch of the model."""
        # print("\n EPOCH \n")

        self.model.train()  # Set the model to training mode
        train_loss = 0.0

        # Training loop
        for self.train_batch_idx, batch in enumerate(self.train_dataloader):

            batch = self.prepare_batch(batch)

            self.optim.zero_grad()  # Reset gradients
            loss = self.model.training_step(batch)  # Compute loss
            loss.backward()  # Backpropagation

            self.optim.step()  # Update model parameters
            train_loss += loss.item()

        avg_train_loss = train_loss / self.num_train_batches
        # print(f"Epoch {self.epoch + 1}/{self.max_epochs}, Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        if self.num_val_batches > 0:
            self.model.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():  # Disable gradient computation for validation
                for self.val_batch_idx, batch in enumerate(self.val_dataloader):
                    batch = self.prepare_batch(
                        batch
                    )  # Move batch to the correct device
                    loss = self.model.validation_step(batch)  # Compute validation loss
                    val_loss += loss.item()

            avg_val_loss = val_loss / self.num_val_batches
            # print(f"Epoch {self.epoch + 1}/{self.max_epochs}, Validation Loss: {avg_val_loss:.4f}")
        else:
            avg_val_loss = None
            val_loss = None

        # Update metadata with training step information
        self.metadata["training_epochs"].append(
            {
                "epoch": self.epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
            }
        )

        self.print_training_status()

    def prepare_batch(self, batch):
        # Try sending the batch to the GPU if possible
        batch = [b.to(try_gpu()) for b in batch]
        return batch

    def print_training_status(self):
        train_loss = self.metadata["training_epochs"][-1]["avg_train_loss"]
        val_loss = self.metadata["training_epochs"][-1]["avg_val_loss"] 
        val_loss = val_loss if val_loss is not None else np.NaN

        end = "\n" if self.epoch == self.max_epochs -1 else "\r"  # Carriage return unless last epoch
        print(f"Epoch {self.epoch + 1}/{self.max_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", end=end)


def try_gpu():
    """Return gpu if exists, otherwise return cpu"""
    if torch.cuda.device_count() >= 1:
        return torch.device("cuda")
    return torch.device("cpu")
