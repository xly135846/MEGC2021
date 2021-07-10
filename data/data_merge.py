import torch 
from torch.utils.data.dataset import Dataset

import random
import numpy as np

class merge_data(Dataset):
    def __init__(self, path, subjects, labels, length, batch_size, data_transforms=None, mode="Train", express="micro"):
        self.img_list = []
        self.label_list = []
        self.data_transforms  = data_transforms
        self.inter_length = length
        self.batch_size = batch_size

        for subject in range(len(subjects)):
            temp = np.load(path+"/"+subjects[subject]+"_"+express+".npy", allow_pickle=True)
            for i in range(int(len(temp)-self.inter_length)):
                self.img_list.append(temp[i:i+self.inter_length])

                IOU_list = []
                for label in labels[subject]:
                    IOU_list.append(self.cal_IOU([i*4, (i+self.inter_length)*4], label))
                IOU_list = np.array(IOU_list)
                if len(np.where(IOU_list>=0.5)[0])>0:
                    self.label_list.append(1)
                else:
                    self.label_list.append(0)

        # 重复采样
        img_list_array = np.array(self.img_list)
        label_list_array = np.array(self.label_list)
        index_0 = np.where(label_list_array==0)[0]
        index_1 = np.where(label_list_array>=1)[0]
        print("index_0:",len(index_0),"index_1", len(index_1))

        random_sample_index = np.random.choice(index_0, int(len(index_1)*1)).tolist()

        random_sample_img = list(img_list_array[random_sample_index])
        random_sample_label = list(label_list_array[random_sample_index])
        sample_img_1 = list(img_list_array[index_1])
        sample_label_1 = list(label_list_array[index_1])

        self.img_list = sample_img_1+random_sample_img
        self.label_list = sample_label_1+random_sample_label

        self.img_list = self.img_list[:int(len(self.img_list)/self.batch_size)*self.batch_size]
        self.label_list = self.label_list[:int(len(self.label_list)/self.batch_size)*self.batch_size]
        self.length = len(self.img_list)

    def __len__(self):
        return self.length

    def cal_IOU(self, interval_1, interval_2):
        intersection = [max(interval_1[0], interval_2[0]), min(interval_1[1], interval_2[1])]
        union_set    = [min(interval_1[0], interval_2[0]), max(interval_1[1], interval_2[1])]
        if intersection[0]<=intersection[1]:
            len_inter = intersection[1]-intersection[0]+1
            len_union = union_set[1]-union_set[0]+1
            return len_inter/(interval_1[1]-interval_1[0]+1)
        else:
            return 0

    def normal_arr(self, arr):
        result = (arr-np.min(arr))/(np.max(arr)- np.min(arr))
        return result

    def __getitem__(self, index):
        img  = self.img_list[index]
        label = self.label_list[index]

        final = np.zeros((3*self.inter_length, 42, 42))
        for i in range(len(img)):
            img[i][:,:,0] = self.normal_arr(img[i][:,:,0])
            img[i][:,:,1] = self.normal_arr(img[i][:,:,1])
            img[i][:,:,2] = self.normal_arr(img[i][:,:,2])
            final[i*3:(i+1)*3] = np.transpose(img[i], (2,0,1))

        return torch.FloatTensor(final), torch.LongTensor([label])

