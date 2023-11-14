import numpy as np
import torch
from networks import *
import torch.nn as nn
# - Matplotlib
import matplotlib.pyplot as plt
# - Rockpool time-series handling
from rockpool import TSEvent, TSContinuous
import torch
import samna
# - Disable warnings
import pickle
import warnings
warnings.filterwarnings('ignore')
# - Import the computational modules and combinators required for the networl
from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.nn.combinators import Sequential, Residual
from rockpool.transform import quantize_methods as q
from torch.utils.data import Dataset,random_split,DataLoader
from pathlib import Path
from tqdm.auto import tqdm
from rockpool.devices import xylo as x
from rockpool.devices.xylo import imu 
from rockpool.nn.networks import SynNet,WaveSenseNet    
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from rockpool.nn.combinators import Sequential, Residual
from rockpool.nn.modules.torch.lif_torch import PeriodicExponential
from rockpool.nn.modules import LIFBitshiftTorch
from parameters import *
from function import *
from train import *
from synnet_rebuild import synnet_mix_rebuild



model = synnet_mix_rebuild
model.load("/home/ruixing/workspace/bc_interface/models/LIFTorch_lack_of_Linear_91channel_modelmix_spike_624_0.9020618556701031.pth")

# load data 
with open('/home/ruixing/workspace/bc_interface/dataset/train_dataset_91.pkl','rb') as file:
    train_dataset = pickle.load(file)
with open('/home/ruixing/workspace/bc_interface/dataset/test_dataset_91.pkl','rb') as file:
    test_dataset = pickle.load(file)

train_dataloader  = DataLoader(train_dataset,batch_size=batch_size,drop_last=True)
test_dataloader = DataLoader(test_dataset,batch_size=len(test_dataset))
# give network LinearTorch with random weights
new_model = new_synnet_mix_rebuild 

def add_linear_layer(new_model, original_model):
    for i, m in enumerate(new_model.seq):
        if isinstance(m, LinearTorch):
            if i == 0:
                pass
            else: new_model.seq[i].weight.data = original_model.seq[i-1].weight.data
        if isinstance(m, LIFTorch):
            new_model.seq[i].tau_syn.data = original_model.seq[i-1].tau_syn.data
    return new_model
complete_model = add_linear_layer(new_model, model)


# - Import the Xylo HDK detection function
from rockpool.devices.xylo import find_xylo_hdks
import samna
connected_hdks, support_modules, chip_versions = find_xylo_hdks()
found_xylo = len(connected_hdks) > 0
if found_xylo:
    hdk = connected_hdks[0]
    x = support_modules[0]
else:
    assert False, 'This tutorial requires a connected Xylo HDK to run.'
spec = x.mapper(complete_model.as_graph(), weight_dtype = 'float')
spec.update(q.global_quantize(**spec))


graph = samna.graph.EventFilterGraph()
_, etf, state_buf = graph.sequential([hdk.get_model_source_node(), "XyloImuOutputEventTypeFilter", samna.graph.JitSink()])
etf.set_desired_type('xyloImu::event::Readout')
graph.start()

input_node = samna.graph.source_to(hdk.get_model_sink_node())
input = test_dataset[0][0]
for i in range(91):
    input_node.write([samna.xyloImu.event.WriteMemoryValue(10240+i, input[:,0][i])])
evts = state_buf.get_events()
pass

# - Use rockpool.devices.xylo.config_from_specification
config, is_valid, msg = x.config_from_specification(**spec)
# - Use rockpool.devices.xylo.XyloSamna to deploy to the HDK
if found_xylo:
    modSamna = x.XyloSamna(hdk, config, dt = 0.001)
    print(modSamna)
    




# d = samna.device.get_unopened_devices()
# dk = samna.device.open_device(d[0])
# print(dk)
