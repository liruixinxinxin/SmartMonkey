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
from rockpool.nn.modules.torch.lif_torch import PeriodicExponential
from rockpool.nn.modules import LIFBitshiftTorch
from parameters import *
from function import *
from train import *
from synnet_rebuild import synnet_mix_rebuild

# from samna.flasher import *

# d = get_programmable_devices()[0]
# program_fx3_flash(d, '/home/ruixing/workspace/bc_interface/XyloAv2TestBoard/FX3/motherBoard_0_11_5.img')

# d = get_programmable_devices()[0]
# program_fpga_flash(d, '/home/ruixing/workspace/bc_interface/XyloAv2TestBoard/FPGA/XyloA2TestBoard_1_0_1_1_3.bin')  # you don't need to unplug the board

model = synnet_mix_rebuild
model.load('models/LIFTorch_lack_of_Linear_91channel_modelmix_spike_624_0.9020618556701031.pth')

# load data 
with open('/home/ruixing/workspace/bc_interface/dataset/train_dataset_91.pkl','rb') as file:
    train_dataset = pickle.load(file)
with open('/home/ruixing/workspace/bc_interface/dataset/test_dataset_91.pkl','rb') as file:
    test_dataset = pickle.load(file)

train_dataloader  = DataLoader(train_dataset,batch_size=batch_size,drop_last=True)
test_dataloader = DataLoader(test_dataset,batch_size=len(test_dataset))

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

# - Detect a connected HDK and import the required support package
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
        for batch, target in tqdm(test_dataloader):
            test_targets += target.detach().cpu().numpy().tolist()
            print(f'label:{test_targets}')
            with torch.no_grad():
                batch = batch.to(torch.float32).to(device)
                model.reset_state()
                # out_model, _, rec = model1(batch, record=True)
                # out1 = rec['1_LIFExodus']['spikes']
                batch = np.asarray(batch.cpu().detach().numpy())
                for sample_data in batch:
                    out_model, _, rec = model2(sample_data.astype(np.int16), record=True)
                    out = np.sum(out_model,axis=0)                                       
                    pred = out.argmax(0)
                    print(f'pred:{pred}')
                    test_preds.append(pred)
                    
                    
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
