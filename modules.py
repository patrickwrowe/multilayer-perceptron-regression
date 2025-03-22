import attrs
from torch.nn import Module
from torch import nn, optim


@attrs.define(eq=False)
class LinearReLUMLP(Module):

    # A list of integers specifying the dimensions of the hidden layers
    mlp_arch: list[int]
    learning_rate: float = 0.1
    weight_decay: float = 0.01

    def __attrs_post_init__(self):
        super().__init__()

        # Build the architecture based on the specified layers
        net_arch = []
        for dim in self.mlp_arch:
            net_arch.extend([nn.LazyLinear(dim), nn.ReLU()])
        net_arch.extend([nn.LazyLinear(1)])  # Output linear layer for regression

        self.net = nn.Sequential(*nn.ModuleList(net_arch))

    def apply_initialization(self, inputs):
        self.forward(inputs)
        self.net.apply(self.initialize_parameters)

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        loss_function = nn.MSELoss()
        return loss_function(y_hat, y)

    def configure_optimizers(self):
        # return optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        # Looks like Adam should be the default really
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def initialize_parameters(self, module):

        def initialize_xavier(layer):
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        module.apply(initialize_xavier)

    def training_step(self, batch):
        y_hat = self(*batch[:-1])
        l = self.loss(y_hat, batch[-1])
        return l

    def validation_step(self, batch):
        y_hat = self(*batch[:-1])
        l = self.loss(y_hat, batch[-1])
        return l
