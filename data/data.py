import torch 
from torch.utils.data.dataset import Dataset

import numpy as np

class spot_data(Dataset):
    def __init__(self, path, subjects, dataset='SAMM', express='micro'):
        self.img_list = []
        self.label_list = []
        self.dataset  = dataset

        for subject in range(len(subjects)):
            temp = np.load(path+'/'+subjects[subject]+'_'+express+'.npy', allow_pickle=True)
            self.img_list += list(temp[:,0])
            self.label_list += list(temp[:,1])

        self.length = len(self.img_list)

    def __len__(self):
        return self.length
    
    def normal_arr(self, arr):
        result = (arr-np.min(arr))/(np.max(arr)- np.min(arr))
        return result

    def __getitem__(self, index):
        img  = self.img_list[index]
        label = self.label_list[index]

        input_1 = np.expand_dims(img[:,:,0], axis=0)
        input_2 = np.expand_dims(img[:,:,1], axis=0)
        input_3 = np.expand_dims(img[:,:,2], axis=0)

        if self.dataset == 'CAS':
            input_1 = self.normal_arr(input_1)
            input_2 = self.normal_arr(input_2)
            input_3 = self.normal_arr(input_3)        

        return torch.FloatTensor([input_1, input_2, input_3]), torch.FloatTensor([label])

