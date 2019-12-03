# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 11:16:39 2019

@author: hongj
"""

import dataset
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
from model import CharRNN, CharLSTM
import multiprocessing
import time
import argparse
import numpy as np
import pickle


def train(model, trn_loader, device, criterion, optimizer):
    
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """

    total_loss= 0
    batch_size= trn_loader.batch_size
    
    model.train()
    
    for num_batch_iter, (x, target) in enumerate(trn_loader):
        
        batch_size= x.shape[0]
        x, target= x.to(device), target.to(device)
        target= target.contiguous().view(-1,1).squeeze(-1)
        h= model.init_hidden(batch_size)
        
        optimizer.zero_grad()
        model_output, h= model(x, h)
        
        batch_loss= criterion(model_output, target)
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
                
        total_loss+= batch_loss.item()
        
    trn_loss= total_loss/ (num_batch_iter+ 1)

    return trn_loss


def validate(model, val_loader, device, criterion):
    
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    total_loss= 0
    batch_size= val_loader.batch_size
    
    model.eval()
    
    for num_batch_iter, (x, target) in enumerate(val_loader):
                
        batch_size= x.shape[0]
        x, target= x.to(device), target.to(device)
        target= target.contiguous().view(-1,1).squeeze(-1)
#        h= tuple([e.data for e in h])
        h= model.init_hidden(batch_size)
        
        model_output, h= model(x, h)
        
        total_loss+= criterion(model_output, target).item()
        
    val_loss= total_loss/ (num_batch_iter+ 1)

    return val_loss


def main():
    
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    
    parser= argparse.ArgumentParser()
    parser.add_argument('--val_ratio', type= float, default= .5, help= 'The ratio for valid set')
    parser.add_argument('--n_layers', type= int, default=4, help= 'Number of stacked RNN layers')
    parser.add_argument('--n_hidden', type= int, default= 512, help= 'Number of hidden neurons of RNN cells')
    parser.add_argument('--drop_prob', type= float, default= 0.1, help= 'Dropout probability')
    parser.add_argument('--num_epochs', type= int, default= 100, help= 'The number of epochs')
    parser.add_argument('--lr', type= float, default= 0.001, help= 'Learning rate')
    parser.add_argument('--T', type= float, default= 1., help= 'The temperature parameter for softmax function')
    parser.add_argument('--device', type= str, default= 'gpu', help= 'For cpu: \'cpu\', for gpu: \'gpu\'')
    parser.add_argument('--batch_size', type= int, default= 256, help= 'Size of batches for training')
    parser.add_argument('--save_dir', type= str, default= '../model', help= 'Name of saved model.')
    parser.add_argument('--rnn', type= bool, default= True, help= 'Train vanilla rnn model')
    parser.add_argument('--lstm', type= bool, default= True, help= 'Train lstm model')
    parser.add_argument('--chunk_size', type= int, default= 30, help= 'Chunk size(sequence length)')
    parser.add_argument('--s_step', type= int, default= 5, help= 'Sequence step')

    args = parser.parse_args()

    n_cpu= multiprocessing.cpu_count()
        
    if args.device== 'gpu': 
        args.device= 'cuda'
    device= torch.device(args.device)
        
    chunk_size= args.chunk_size
    s_step= args.s_step
    num_epochs= args.num_epochs
    batch_size= args.batch_size
    val_ratio= args.val_ratio
    shuffle_dataset= True
    random_seed= 42
    
    datasets= dataset.Shakespeare('shakespeare_train.txt', chunk_size, s_step)
    
    dataset_size= len(datasets)
    indices= list(range(dataset_size))
    split= int(np.floor(val_ratio* dataset_size))
    
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
    train_indices, val_indices= indices[split:], indices[:split]

    train_sampler= SubsetRandomSampler(train_indices)
    valid_sampler= SubsetRandomSampler(val_indices)

    trn_loader= DataLoader(datasets, batch_size= batch_size, 
                           sampler= train_sampler, num_workers= n_cpu)
    val_loader= DataLoader(datasets, batch_size= batch_size,
                           sampler= valid_sampler, num_workers= n_cpu)
    
    chars= datasets.chars
    
    print('-----Train Vanilla RNN Model-----')
    
    if args.rnn:
        
        model= CharRNN(chars, args).to(device)
        optimizer= Adam(model.parameters(), lr= args.lr)
        criterion= nn.CrossEntropyLoss()
        
        rnn_trn_loss, rnn_val_loss= [], []
        best_val_loss= np.inf
        
        for epoch in range(args.num_epochs):
    
            epoch_time= time.time()
            
            trn_loss= train(model, trn_loader, device, criterion, optimizer)
            val_loss= validate(model, val_loader, device, criterion)
            
            rnn_trn_loss.append(trn_loss)
            rnn_val_loss.append(val_loss)
            
            print('Epoch: %3s/%s...'%(epoch+ 1, num_epochs),
                  'Train Loss: %.4f...'%trn_loss,
                  'Val Loss: %.4f...'%val_loss,
                  'Time: %.4f'%(time.time()- epoch_time))
            
            if val_loss< best_val_loss:
                best_val_loss= val_loss
                torch.save(model.state_dict(), '%s/rnn_T_%s.pth'%(args.save_dir, str(args.T).replace('.', '_')))
                
        with open('../results/rnn_loss.pkl', 'wb') as f:
            pickle.dump((rnn_trn_loss, rnn_val_loss), f)
        
    print('-----Train LSTM Model-----')    
    
    if args.lstm:
        
        model= CharLSTM(chars, args).to(device)
        optimizer= Adam(model.parameters(), lr= args.lr)
        criterion= nn.CrossEntropyLoss()
        
        lstm_trn_loss, lstm_val_loss= [], []
        best_val_loss= np.inf
        
        for epoch in range(args.num_epochs):
    
            epoch_time= time.time()
            
            trn_loss= train(model, trn_loader, device, criterion, optimizer)
            val_loss= validate(model, val_loader, device, criterion)
            
            lstm_trn_loss.append(trn_loss)
            lstm_val_loss.append(val_loss)
            
            print('Epoch: %3s/%s...'%(epoch+ 1, num_epochs),
                  'Train Loss: %.4f...'%trn_loss,
                  'Val Loss: %.4f...'%val_loss,
                  'Time: %.4f'%(time.time()- epoch_time))
            
            if val_loss< best_val_loss:
                best_val_loss= val_loss
                torch.save(model.state_dict(), '%s/lstm_T_%s.pth'%(args.save_dir, str(args.T).replace('.', '_')))
                
        with open('../results/lstm_loss.pkl', 'wb') as f:
            pickle.dump((lstm_trn_loss, lstm_val_loss), f)
            
        
if __name__ == '__main__':
    
    main()
