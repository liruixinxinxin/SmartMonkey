## Imports

# %matplotlib widget # uncomment for interactive plots
from nlb_tools.nwb_interface import NWBDataset
import numpy as np
import pickle
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from mydataset import Mydataset
plt.style.use('seaborn')
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

label = {
        'class0': [0, 7, 33, 94],
        'class1': [1, 10, 43],
        'class2': [3, 14, 70, 84],
        'class3': [5, 23, 55, 80, 83],
        'class4': [6, 22, 54, 106],
        'class5': [12, 26,29, 72],
        'class6': [9, 61, 82],
        'class7': [17, 75, 86],
        }


# label = {
#         'class0': [0],
#         'class1': [1],
#         'class2': [3],
#         'class3': [5],
#         'class4': [6],
#         'class5': [12],
#         'class6': [9],
#         'class7': [17],
#         }

with open('/home/liruixin/workspace/bcinterface/dataset/My_dataset.pkl','rb') as file:
    dataset = pickle.load(file)
    

# dataset = NWBDataset("/home/ruixing/workspace/bcinterface/000128/sub-Jenkins", "*train", split_heldout=False)
conds = dataset.trial_info.set_index(['trial_type', 'trial_version']).index.unique().tolist()
data_list = []
label_list = []
for index in tqdm(range(len(conds))):
    cond = conds[index]
    mask = np.all(dataset.trial_info[['trial_type', 'trial_version']] == cond, axis=1)
    # Extract trial data
    trial_data = dataset.make_trial_data(align_field='move_onset_time', align_range=(-50, 450), ignored_trials=(~mask))
    begin = 0
    end = begin + 500
    for key, values_list in label.items():
        for trial in (range(trial_data.shape[0]//500)):
            if index in values_list:
                sample = trial_data['spikes'][begin:end].to_numpy()
                # spilt sample to 14 (500,13)
                # split_arrays = np.array_split(sample, 14, axis=1)
                # summed_arrays = [np.sum(split, axis=1).reshape(-1, 1) for split in split_arrays]
                # result_array = np.hstack(summed_arrays)
                data_list.append(sample)
                label_list.append(int(key[-1]))
                begin += 500
                end += 500


random.seed(42)  
combined = list(zip(data_list, label_list))
random.shuffle(combined)
data_list, label_list = zip(*combined)
data_list = list(data_list)
label_list = list(label_list)


train_data, test_data, train_labels, test_labels = train_test_split(
                                                    data_list, 
                                                    label_list, 
                                                    test_size=0.3, 
                                                    random_state=43)

train_dataset = Mydataset(train_data,train_labels)
test_dataset = Mydataset(test_data,test_labels)

with open('/home/liruixin/workspace/bcinterface/dataset/train_dataset.pkl','wb') as file:
    pickle.dump(train_dataset,file)

with open('/home/liruixin/workspace/bcinterface/dataset/test_dataset.pkl','wb') as file:
    pickle.dump(test_dataset,file)