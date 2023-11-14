import torch


num_class = 8
time_step = 500
num_epochs = 2000
num_channel = 182
batch_size = 32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pass