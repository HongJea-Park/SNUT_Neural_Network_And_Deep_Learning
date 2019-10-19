
import dataset
from model import LeNet5, CustomMLP
import tarfile
import os
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import multiprocessing
import pandas as pd


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
        acc: accuracy
    """

    running_loss, correct, total= 0, 0, 0
        
    for num_batch_iter, (x, target) in enumerate(trn_loader):
        
        x, target= x.to(device), target.to(device)
        
        optimizer.zero_grad()
        model_output= model(x)
        
        loss= criterion(model_output, target)
        loss.backward()
        
        optimizer.step()
        
        model_output= model(x)
        _, pred= torch.max(model_output.data, 1)
        
        correct+= (pred== target).sum().item()
        running_loss+= loss.item()
        total+= target.size(0)
        
    trn_loss= running_loss/ (num_batch_iter+ 1)
    acc= correct/ total
    
    return trn_loss, acc


def test(model, tst_loader, device, criterion):
    
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    loss, correct, total= 0, 0, 0
    
    with torch.no_grad():
        
        for num_batch_iter, (x, target) in enumerate(tst_loader):
            
            x, target= x.to(device), target.to(device)
            
            model_output= model(x)
            
            loss+= criterion(model_output, target).item()
            _, pred= torch.max(model_output.data, 1)
            total+= target.size(0)
            correct+= (pred== target).sum().item()
            
    tst_loss= loss/ (num_batch_iter+ 1)
    acc= correct/ total

    return tst_loss, acc


def main():
    
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    
    n_cpu= multiprocessing.cpu_count()
    
    data_dir_dict= {'train': '../data/train', 'test': '../data/test'}
    for _, data_dir in data_dir_dict.items():
        
        if not os.path.isdir(data_dir):
            
            tar= tarfile.open('%s.tar'%data_dir, mode= 'r')
            tar.extractall(path= './data/')

    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net= LeNet5().to(device)
    mlp= CustomMLP().to(device)
        
    criterion= nn.CrossEntropyLoss()
    net_optimizer= SGD(net.parameters(), lr= .01, momentum= .9)
    mlp_optimizer= SGD(mlp.parameters(), lr= .01, momentum= .9)
    
    batch_size = 256

    train_dataset= dataset.MNIST(data_dir_dict['train'])
    test_dataset= dataset.MNIST(data_dir_dict['test'])
    
    train_loader= DataLoader(dataset= train_dataset, 
                              batch_size= batch_size,
                              shuffle= True,
                              num_workers= n_cpu)
    
    test_loader= DataLoader(dataset= test_dataset,
                            batch_size= batch_size,
                            shuffle= False, 
                            num_workers= n_cpu)
    
    num_epochs= 30
    
    net_trn_loss_list, net_trn_acc_list, net_tst_loss_list, net_tst_acc_list= [], [], [], []
    
    print('--------------LeNet5--------------')
    for epoch in range(num_epochs):
        
        net_time= time.time()
        
        net_trn_loss, net_trn_acc= train(net, train_loader, device, criterion, net_optimizer)
        net_tst_loss, net_tst_acc= test(net, test_loader, device, criterion)
        
        print('%s epoch || %.5f time'%((epoch+ 1), time.time()- net_time))
        print('avg train loss: %.5f | avg train acc: %.5f ||| avg test loss: %.5f | avg test acc: %.5f'\
              %(net_trn_loss, net_trn_acc, net_tst_loss, net_tst_acc))
        print('')
        
        net_trn_loss_list.append(net_trn_loss)
        net_trn_acc_list.append(net_trn_acc)
        net_tst_loss_list.append(net_tst_loss)
        net_tst_acc_list.append(net_tst_acc)
    
    mlp_trn_loss_list, mlp_trn_acc_list, mlp_tst_loss_list, mlp_tst_acc_list= [], [], [], []
    
    print('--------------CustomMLP--------------')        
    for epoch in range(num_epochs):
        
        mlp_time= time.time()
        
        mlp_trn_loss, mlp_trn_acc= train(mlp, train_loader, device, criterion, mlp_optimizer)
        mlp_tst_loss, mlp_tst_acc= test(mlp, test_loader, device, criterion)

        print('%s epoch || %.5f time'%((epoch+ 1), time.time()- mlp_time))
        print('avg train loss: %.5f | avg train acc: %.5f ||| avg test loss: %.5f | avg test acc: %.5f'\
              %(mlp_trn_loss, mlp_trn_acc, mlp_tst_loss, mlp_tst_acc))
        print('')
        
        mlp_trn_loss_list.append(mlp_trn_loss)
        mlp_trn_acc_list.append(mlp_trn_acc)
        mlp_tst_loss_list.append(mlp_tst_loss)
        mlp_tst_acc_list.append(mlp_tst_acc)
        
    net_result, mlp_result= {}, {}
    
    net_result['LeNet5 Train Loss']= net_trn_loss_list
    net_result['LeNet5 Train Accuracy']= net_trn_acc_list
    net_result['LeNet5 Test Loss']= net_tst_loss_list
    net_result['LeNet5 Test Accuracy']= net_tst_acc_list
    
    mlp_result['MLP Train Loss']= mlp_trn_loss_list
    mlp_result['MLP Train Accuracy']= mlp_trn_acc_list
    mlp_result['MLP Test Loss']= mlp_tst_loss_list
    mlp_result['MLP Test Accuracy']= mlp_tst_acc_list
    
    net_result= pd.DataFrame(net_result, columns= ['LeNet5 Train Loss', 'LeNet5 Train Accuracy',
                                                   'LeNet5 Test Loss', 'LeNet5 Test Accuracy'])
    
    mlp_result= pd.DataFrame(mlp_result, columns= ['MLP Train Loss', 'MLP Train Accuracy',
                                                   'MLP Test Loss', 'MLP Test Accuracy'])
    
    if not os.path.exists('../results/'):
        os.makedirs('../results/')
    
    net_result.to_csv('../results/LeNet5.csv', index= False)
    mlp_result.to_csv('../results/MLP.csv', index= False)
    

if __name__ == '__main__':
    main()
