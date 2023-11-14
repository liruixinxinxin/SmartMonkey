import pickle

from parameters import *
from networks import *
from torch.utils.data import DataLoader
from train import *
from synnet_rebuild import synnet_mix_rebuild

# load data
with open('/home/liruixin/workspace/bcinterface/dataset/train_dataset.pkl','rb') as file:
    train_dataset = pickle.load(file)
with open('/home/liruixin/workspace/bcinterface/dataset/test_dataset.pkl','rb') as file:
    test_dataset = pickle.load(file)

train_dataloader  = DataLoader(train_dataset,batch_size=batch_size,drop_last=True)
test_dataloader = DataLoader(test_dataset,batch_size=len(test_dataset))

#------------------------------#
# transform 182 to 16 channels #
#------------------------------#
# trans_model = transform_synnet

# load network

model = synnet_mix

# training

# ann_train(device=device,
#           train_dataloader=train_dataloader,
#           test_dataloader = test_dataloader,
#           model=ann)






snn_train_spike(device=device,
               train_dataloader=train_dataloader,
               test_dataloader=test_dataloader,
               model=model)



# snn_train_spike_mix(device=device,
#                train_dataloader=train_dataloader,
#                test_dataloader=test_dataloader,
#                model1=transform_synnet,
#                model2=model2
#                )
