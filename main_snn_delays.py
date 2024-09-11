from datasets import DVS_dataloaders
from config import Config
from snn_delays import SnnDelays
import torch
from config import *
from spikingjelly.activation_based import functional, neuron

train_loader, valid_loader = DVS_dataloaders(config)
test_loader = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SnnDelays(config).to(device)

size_model = 0
for param in model.parameters():
    if param.data.is_floating_point():
        size_model += param.numel() * torch.finfo(param.data.dtype).bits
    else:
        size_model += param.numel() * torch.iinfo(param.data.dtype).bits
print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")
#functional.set_backend(model, 'cupy', instance=neuron.LIFNode)
model.train_model(train_loader, valid_loader, test_loader, device)