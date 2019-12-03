# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:45:59 2019

@author: hongj
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

    
class Shakespeare(Dataset):
    
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1)  Load input file and construct character dictionary {index: character}.
            You need this dictionary to generate characters.
            
        2)  Make list of character indices using the dictionary.
    
        3)  Split the data into chunks of sequence length 30.
            You should create targets appropriately.
        
    """

    def __init__(self, input_file):

        # write your codes here
        self.chunk_size= 30
        
        with open ('./data/%s'%input_file, 'r') as f:
            self.text= f.read()
        
        chars= set(self.text)
        chars.add('eos')
        chars= tuple(chars)
        
        self.n_labels= len(chars)
        self.int2char= dict(enumerate(chars))
        self.char2int= {c: i for i, c in self.int2char.items()}
        
        self.encoded= np.array([self.char2int[c] for c in self.text])
    
        
    def __len__(self):
        
        return len(self.encoded)- self.chunk_size


    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx= idx.tolist()
            
        input_array= np.array([self.char2int[c] for c in self.text[idx: idx+ self.chunk_size]])
        target_array= np.array([self.char2int[c] for c in self.text[idx+ 1: idx+ self.chunk_size]])
        target_array= np.append(target_array, self.char2int['eos'])
        
        input= self.one_hot_encode(input_array)
        target= self.one_hot_encode(target_array)
        
        return input, target
    
    
    def one_hot_encode(self, arr):
        
        one_hot= np.zeros((arr.shape[0], self.n_labels), dtype= np.float32)
        one_hot[np.arange(one_hot.shape[0]), arr.flatten()]= 1.
        one_hot= one_hot.reshape((*arr.shape, self.n_labels))
        
        return one_hot

    
if __name__ == '__main__':

    # write test codes to verify your implementations

    train_dataset= Shakespeare('shakespeare_train.txt')

    train_loader= DataLoader(dataset= train_dataset,
                             batch_size= 1,
                             shuffle= True)
    
    for batch_idx, (x, target) in enumerate(train_loader):
        
        if batch_idx% 10== 0:
            print(x, target)
            
    
    