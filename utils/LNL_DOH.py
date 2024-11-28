import numpy as np
import pandas as pd
import torch
import random
from PIL import Image
import pywt
import torch.nn.functional as  F


class Data_Process():
    #print(data)
    def __init__(self, data, train=True, transform=None, target_transform=None, 
                 noise_type=None, INCV_b = 0.2, INCV_c = 0, random_state=0 ):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        
        self.noise_type=noise_type
        
        if self.train:
            self.train_data = torch.tensor(data.iloc[:, :120].values)
            self.train_labels = torch.tensor(data[['Label']].values)
            # 计算需要填充的长度
            original_length = self.train_data.shape[1]
            target_length = 120
            padding_length = target_length - original_length

            # 在最后一个维度上填充0
            padded_data = F.pad(self.train_data, (0, padding_length), "constant", 0)
            self.train_data = padded_data.reshape(-1,1,120)

            df_train_YY_noise =  data[['Label']].copy()

            
            YY_counts = data[['Label']].value_counts(sort=False).tolist()
            YY_change = np.zeros(len(YY_counts))
            for i in range(len(YY_counts)):
                if i == 0:
                    YY_change[i] = int(YY_counts[i]*INCV_b)
                else:
                    YY_change[i] = int(YY_counts[i]*INCV_c)
            
            ####
            rslt = []
            index = []
            
            for i in range(len(YY_counts)):
                rslt.append(df_train_YY_noise[df_train_YY_noise['Label-noise'] == i].index.tolist())
                index.append(random.sample(rslt[i], int(YY_change[i])))
    
            for i in range(len(YY_counts)):
                if i == 0 :
                    for idx in index[i]:
                        # 假设 len(YY_counts) 是你数组的长度
                        length = len(YY_counts)

                        # 创建一个包含所有可能数字的列表，除了0
                        choices = [i for i in range(length) if i != 0]

                        random_choice = random.choice(choices)
                        df_train_YY_noise.loc[idx,"Label-noise"] = random_choice
                else:
                    for idx in index[i]:
                        df_train_YY_noise.loc[idx,"Label-noise"] = 0
            
            
            self.train_noisy_labels = torch.tensor(df_train_YY_noise.values)
            self.noise_or_not = torch.tensor([self.train_noisy_labels[i]==self.train_labels[i] 
                                 for i in range(self.train_noisy_labels.shape[0])])
        
        else:
            self.test_data = torch.tensor(data.iloc[:, :120].values)
            # 计算需要填充的长度
            original_length = self.test_data.shape[1]
            target_length = 120
            padding_length = target_length - original_length

            # 在最后一个维度上填充0
            padded_data = F.pad(self.test_data, (0, padding_length), "constant", 0)

            # 改变形状至 [64424, 3, 32, 32]
            self.test_data = padded_data.reshape(-1,1,120)
            self.test_labels = torch.tensor(data[['Label']].values)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_noisy_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.numpy(), mode='L')
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
