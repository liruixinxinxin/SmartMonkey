import torch.nn as nn
from parameters import *
from rockpool.nn.modules import LIFExodus, LIFTorch
from rockpool.nn.networks import SynNet


if str(device) == 'cpu':
    neuron_model = LIFTorch
else:
    neuron_model = LIFExodus

class Myann(nn.Module):
    def __init__(self):
        super(Myann, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 1), stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1416960, 64)
        self.fc2 = nn.Linear(64, num_class)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x.float()))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        # x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        output = self.fc2(x)
        # output = self.softmax(x)
        return output
    
ann = Myann()


synnet = SynNet(
    n_channels=num_channel,
    n_classes=num_class,
    size_hidden_layers = [60],
    output='spikes',
    neuron_model=neuron_model,
    threshold = 0.1,
    dt = 0.001
)


synnet_mix = SynNet(
    n_channels=16,
    n_classes=num_class,
    size_hidden_layers = [16,32],
    time_constants_per_layer = [1,1],
    output='spikes',
    neuron_model=neuron_model,
    threshold = 0.1,
    dt = 0.001
)

synnet1 = SynNet(
    n_channels=num_channel,
    n_classes=16,
    size_hidden_layers = [16],
    time_constants_per_layer = [1],
    output='spikes',
    neuron_model=neuron_model,
    threshold = 0.1,
    dt = 0.001
)

synnet2 = SynNet(
    n_channels=16,
    n_classes=num_class,
    size_hidden_layers = [32],
    time_constants_per_layer = [1],
    output='spikes',
    neuron_model=neuron_model,
    threshold = 0.1,
    dt = 0.001
)



new_synnet_mix_rebuild = SynNet(
    n_channels=16,
    n_classes=num_class,
    size_hidden_layers = [91],
    time_constants_per_layer = [1],
    output='spikes',
    neuron_model=neuron_model,
    threshold = 0.1,
    dt = 0.001
)
pass


