from torch.utils.data import Dataset

class Mydataset(Dataset):
    def __init__(self,data_list,label_list):
        self.data_list = data_list
        self.label_list = label_list
    def __getitem__(self,index):
        return self.data_list[index],self.label_list[index]
    def __len__(self):
        return len(self.data_list)