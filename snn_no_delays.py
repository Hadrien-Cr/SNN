import torch
import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import layer
from spikingjelly.activation_based import functional
from spikingjelly.activation_based import functional, surrogate, neuron

class Net(nn.Module):
    def __init__(self, config):
        super().__init__()

        conv = []
        for i in range(config.n_hidden_layers):
            if conv.__len__() == 0:
                in_channels = config.n_inputs
            else:
                in_channels = config.n_hidden_neurons

            conv.append(layer.Conv2d(in_channels, config.n_hidden_neurons, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(config.n_hidden_neurons))
            conv.append(neuron.LIFNode(surrogate_function=surrogate.ATan(),detach_reset=True))
            conv.append(layer.Dropout(config.dropout_p))
            conv.append(layer.MaxPool2d(2, 2))


        self.conv_cat = nn.Sequential(*conv)
        self.fc = nn.Sequential(
            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(config.n_hidden_neurons * 4 * 4, 512),
            neuron.LIFNode(surrogate_function=surrogate.ATan(),detach_reset=True),

            layer.Dropout(0.5),
            layer.Linear(512, 110),
            neuron.LIFNode(surrogate_function=surrogate.ATan(),detach_reset=True),  
        )
        self.voting_layer = layer.VotingLayer(10)

    def forward(self, x: torch.Tensor):
        x = self.conv_cat(x)
        x = self.fc(x)
        x = self.voting_layer(x)
        return x 
    
    def reset_model(self, train=True):
        functional.reset_net(self)
