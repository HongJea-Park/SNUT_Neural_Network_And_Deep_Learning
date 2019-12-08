# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:45:59 2019

@author: hongj
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch

    
class Shakespeare(Dataset):
    
    """ Shakespeare dataset

    Args:
        input_file: txt file
        chunk_size: sequence length
        s_step: string split step

    """

    def __init__(self, input_file, chunk_size= 30, s_step= 3):
        
        with open ('../data/%s'%input_file, 'r') as f:
            self.text= f.read()

        self.chunk_size= chunk_size
        self.s_step= s_step
        self.chars= tuple(set(self.text))
        self.n_labels= len(self.chars)
        self.int2char= dict(enumerate(self.chars))
        self.char2int= {c: i for i, c in self.int2char.items()}
        
#        self.encoded= np.array([self.char2int[c] for c in self.text])
    
        
    def __len__(self):
        
        return int((len(self.text)- 30)/ self.s_step)


    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx= idx.tolist()
            
        idx*= self.s_step    
        
        input_array= np.array([self.char2int[c] for c in \
                               self.text[idx: idx+ self.chunk_size]])
        target_array= np.array([self.char2int[c] for c in \
                                self.text[idx+ 1: idx+ self.chunk_size+ 1]])
        
        input= torch.Tensor(self.one_hot_encode(input_array))
        target= torch.LongTensor(target_array)
        
        return input, target
    
    
    def one_hot_encode(self, arr):
        
        if type(arr)== int: arr= np.array([arr])
        
        one_hot= np.zeros((arr.shape[0], self.n_labels), dtype= np.float32)
        one_hot[np.arange(one_hot.shape[0]), arr.flatten()]= 1.
        one_hot= one_hot.reshape((*arr.shape, self.n_labels))
        
        return one_hot

    
if __name__ == '__main__':

    dataset= Shakespeare('shakespeare_train.txt', chunk_size= 30, s_step= 3)
    
    batch_size= 512
    validation_split= .5
    shuffle_dataset= False
    random_seed= 42
    
    dataset_size= len(dataset)
    indices= list(range(dataset_size))
    split= int(np.floor(validation_split* dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices= indices[split:], indices[:split]

    train_sampler= SubsetRandomSampler(train_indices)
    valid_sampler= SubsetRandomSampler(val_indices)

    trn_loader= DataLoader(dataset, batch_size= batch_size, 
                           sampler= train_sampler)
    val_loader= DataLoader(dataset, batch_size= batch_size,
                           sampler= valid_sampler)
    
    for batch_idx, (x, target) in enumerate(trn_loader):
        
        if batch_idx% 10== 0:
            print(x, target)
            
    
    