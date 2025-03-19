from typing import Optional
import attrs
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torch.nn import Module


@attrs.define()
class Trainer():

    max_epochs: int
    data: Optional[DataLoader] = None
    model: Optional[Module] = None

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        self.model = model

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
        for batch in self.train_dataloader():
            loss = self.model.training_step(self.prepare_batch(batch))
            
            self.optim.zero_grad_()
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
        # If we end up training with GPUs, send batch to GPU here. 
        device = try_gpu()
        return batch.to(device) 

def try_gpu():
    """Return gpu if exists, otherwise return cpu"""
    if torch.cuda.device_count() >= 1:
        return torch.device('cuda')
    return torch.device('cpu')