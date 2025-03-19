from typing import Optional
import attrs
from pandas.core.dtypes.cast import _maybe_box_and_unbox_datetimelike
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim.optimizer import Optimizer


@attrs.define(slots=False)
class Trainer():

    max_epochs: int

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

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        # Attempt to put the model on the GPU if one is available
        device = try_gpu()
        model.to(device) 
        self.model = model
        print(f"Model running on {device}")

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        assert self.model  # Ensure correct initialization

        self.model.train()  # Put model in training mode
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                self.optim.step()

            self.train_batch_idx += 1

        if self.val_dataloader is None:
            return
        
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1

    def prepare_batch(self, batch):
        # Try sending the batch to the GPU if possible
        batch = [b.to(try_gpu()) for b in batch]
        return batch

def try_gpu():
    """Return gpu if exists, otherwise return cpu"""
    if torch.cuda.device_count() >= 1:
        return torch.device('cuda')
    return torch.device('cpu')