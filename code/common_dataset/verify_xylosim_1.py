# - Numpy
import numpy as np
import torch
from networks import *
import torch.nn as nn
# - Matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams['figure.dpi'] = 300
plt.style.use('seaborn')
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
from tqdm.asyncio import tqdm
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


def change_model(original_model, new_model):
    for i, m in enumerate(new_model.seq):
        if isinstance(m, LinearTorch):
            if i == 3:
                new_model.seq[i].weight.data = torch.zeros_like(original_model.seq[1].tau_syn.data)+1
            if i == 0:
                new_model.seq[i].weight.data = original_model.seq[i].weight.data
        if isinstance(m, LIFExodus):
            if i == 1:
                new_model.seq[i].tau_syn.data = original_model.seq[i].tau_syn.data
    return new_model


def change_model2(original_model, new_model):
    for i, m in enumerate(new_model.seq):
        if isinstance(m, LinearTorch):
            new_model.seq[i].weight.data = original_model.seq[i+3].weight.data
        if isinstance(m, LIFExodus):
            new_model.seq[i].tau_syn.data = original_model.seq[i+3].tau_syn.data
    return new_model



model = synnet_mix
model.load("/home/ruixing/workspace/bc_interface/models/lack_of_Linear_91channel_modelmix_spike_624_0.9020618556701031.pth")
transf_synnet1 = change_model(synnet_mix, synnet1)
transf_synnet2 = change_model2(synnet_mix, synnet2)

g = transf_synnet2.as_graph()
spec = imu.mapper(g, weight_dtype='float', threshold_dtype='float', dash_dtype='float')
quant_spec = spec.copy()
# - Quantize the specification
spec.update(q.global_quantize(**spec))
# - Use rockpool.devices.xylo.config_from_specification
config, is_valid, msg = imu.config_from_specification(**spec)
modSim = imu.XyloSim.from_config(config)


# load data
with open('/home/liruixin/workspace/bcinterface/dataset/train_dataset_91.pkl','rb') as file:
    train_dataset = pickle.load(file)
with open('/home/liruixin/workspace/bcinterface/dataset/test_dataset_91.pkl','rb') as file:
    test_dataset = pickle.load(file)

train_dataloader  = DataLoader(train_dataset,batch_size=batch_size,drop_last=True)
test_dataloader = DataLoader(test_dataset,batch_size=len(test_dataset))

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
            with torch.no_grad():
                batch = batch.to(torch.float32).to(device)
                model.reset_state()
                out_model, _, rec = model1(batch, record=True)
                out1 = rec['1_LIFExodus']['spikes']
                out1 = np.asarray(out1.cpu().detach().numpy())
                for sample_data in out1:
                    out_model, _, rec = model2(sample_data, record=True)
                    out = np.sum(out_model,axis=0)  
                    pred = out.argmax(0)
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
        model2= modSim)