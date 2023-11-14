import numpy as np
import torch
from networks import *
import torch.nn as nn
# - Matplotlib
import matplotlib.pyplot as plt
# - Rockpool time-series handling
from rockpool import TSEvent, TSContinuous
import torch
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
from rockpool.devices.xylo.syns61201 import xa2_devkit_utils as hdu
import time




model = synnet_mix_rebuild
model.load('/home/ruixing/workspace/bc_interface/models/LIFTorch_lack_of_Linear_91channel_modelmix_spike_624_0.9020618556701031.pth')

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



# - Use rockpool.devices.xylo.config_from_specification
config, is_valid, msg = x.config_from_specification(**spec)
# - Use rockpool.devices.xylo.XyloSamna to deploy to the HDK
if found_xylo:
    modSamna = x.XyloSamna(hdk, config, dt = 0.001)
    print(modSamna)
    
train_dataloader  = DataLoader(train_dataset,batch_size=batch_size,drop_last=True)
test_dataloader = DataLoader(test_dataset,batch_size=len(test_dataset))

print(f'Clock freq. set to {hdu.set_xylo_core_clock_freq(modSamna._device, 12.5)} MHz')

def snn_train_spike(device, train_dataloader, test_dataloader, model1, model2):
    model1.to(device)
    print('device:',device)
    losslist = []
    accuracy = []
    f1s = []
    precision = []
    recall = []
    cmlist= []
    for epoch in range(1):
        # scheduler.step()
        test_preds = []
        test_targets = []
        io_power_list = []
        core_power_list = []
        for batch, target in tqdm(test_dataloader):
            test_targets += target.detach().cpu().numpy().tolist()
            print(test_targets)
            with torch.no_grad():
                batch = batch.to(torch.float32).to(device)
                model.reset_state()
                # out_model, _, rec = model1(batch, record=True)
                # out1 = rec['1_LIFExodus']['spikes']
                batch = np.asarray(batch.cpu().detach().numpy())
                                    
                delay_list = []
                for n, sample_data in enumerate(batch):
                    result = []
                    data_array = np.zeros((8, 1)).astype(np.int16)
                    for timestep_data in sample_data:
                        out_model, _, rec = model2(timestep_data.astype(np.int16).reshape(1,-1), record=False, record_power=False)
                        # io_power = np.mean(rec['io_power'])
                        # core_power = np.mean(rec['core_power'])
                        # print('io_power:',io_power,'core_power:',core_power)
                        # io_power_list.append(io_power)
                        # core_power_list.append(core_power)
                        
                        data_array = np.concatenate((data_array,out_model.reshape(8,1)), axis=1)
                        out = np.sum(data_array,axis=1)  
                        pred = out.argmax(0)
                        result.append(pred)
                        if pred == test_targets[n]:
                            # print(f'{n}:fuond true')
                            appear,index = find_first_consecutive(result, test_targets[n])
                            if appear == 1:
                                delay = 0.001*index
                                delay_list.append(delay)
                                break
                        elif len(result)==500:
                            delay_list.append(0.5)
                        if pred != test_targets[n]:
                            pass
                            # print(f'{n}:fuond false,{pred},{test_targets[n]}')
                        test_preds.append(pred)
                        
                    print(delay_list)      

        f1 = f1_score(test_targets, test_preds, average="macro")
        _, test_precision, test_recall, _ = precision_recall_fscore_support(
            test_targets, test_preds, labels=np.arange(8)
        )
        test_accuracy = accuracy_score(test_targets, test_preds)
        cm = confusion_matrix(test_targets, test_preds)
        # losslist.append(test_loss)
        f1s.append(f1)
        precision.append(test_precision)
        recall.append(test_recall)
        accuracy.append(test_accuracy)
        cmlist.append(cm)
        print(f"F1 Score = {f1}")
        print(f"Val Precision = {test_precision}, Recall = {test_recall}")
        print_colorful_text(f"Accuracy:{test_accuracy}", 'yellow')
        print("Confusion Matrix:")
        print(cm)
        
        
snn_train_spike(device= device,
        train_dataloader=train_dataset,
        test_dataloader= test_dataloader,
        model1= synnet1,
        model2= modSamna)
