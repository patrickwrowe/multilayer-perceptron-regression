import torch
import attrs
from torch.nn import Module
from torch import nn, optim

@attrs.define
class LinearReLUMLP(Module):

    net: nn.Module

    # A list of integers specifying the dimensions of the hidden layers
    mlp_arch: list[int]
    learning_rate: float = 0.01

    def __attrs_post_init__(self):

        # Build the architecture based on the specified layers
        net_arch = []
        for dim in self.mlp_arch:
            net_arch.extend([nn.LazyLinear(dim), nn.ReLU()])
        net_arch.extend([nn.LazyLinear(1)]) # Output linear layer for regression

        self.net = nn.ModuleList(net_arch)

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        loss_function = nn.MSELoss()
        return loss_function(y_hat, y)

    def configure_optimizers(self):
        optimiser = optim.SGD(self.parameters(), self.learning_rate)
        return optimiser

    def initialize_parameters(self, module):

        def initialize_xavier(layer):
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))
        
        module.apply(initialize_xavier)

