# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 21:07:18 2019

@author: hongj
"""

import pandas as pd
import matplotlib.pyplot as plt


if __name__== '__main__':
    
    net= pd.read_csv('../results/LeNet5.csv')
    mlp= pd.read_csv('../results/MLP.csv')
    
    net_trn_loss, net_trn_acc= net['LeNet5 Train Loss'], net['LeNet5 Train Accuracy']
    net_tst_loss, net_tst_acc= net['LeNet5 Test Loss'], net['LeNet5 Test Accuracy']
    
    mlp_trn_loss, mlp_trn_acc= mlp['MLP Train Loss'], mlp['MLP Train Accuracy']
    mlp_tst_loss, mlp_tst_acc= mlp['MLP Test Loss'], mlp['MLP Test Accuracy']
    
    idx= net.index
    
    fig= plt.figure(figsize= (9, 9))
    fig.suptitle('', fontsize= 16)
    #plt.subplots_adjust(wspace= .5, hspace= 1.)
    
    ax1= fig.add_subplot(2, 2, 1)
    ax1.plot(idx, net_trn_loss, 'r', label= 'LeNet5 Train Loss')
    ax1.plot(idx, mlp_trn_loss, 'g', label= 'MLP Train Loss')
    ax1.legend(loc= 'upper right')
    ax1.set_ylim(top= 1.)
    ax1.set_title('Train Loss Curve LeNet5 vs MLP')
    
    ax2= fig.add_subplot(2, 2, 2)
    ax2.plot(idx, net_tst_loss, 'r', label= 'LeNet5 Test Loss')
    ax2.plot(idx, mlp_tst_loss, 'g', label= 'MLP Test Loss')
    ax2.legend(loc= 'upper right')
    ax2.set_title('Test Loss Curve LeNet5 vs MLP')
    
    ax3= fig.add_subplot(2, 2, 3)
    ax3.plot(idx, net_trn_acc, 'r', label= 'LeNet5 Train Accuracy')
    ax3.plot(idx, mlp_trn_acc, 'g', label= 'MLP Train Accuracy')
    ax3.legend(loc= 'lower right')
    ax3.set_ylim(bottom= .7)
    ax3.set_title('Train Accuracy Curve LeNet5 vs MLP')
    
    ax4= fig.add_subplot(2, 2, 4)
    ax4.plot(idx, net_tst_acc, 'r', label= 'LeNet5 Test Accuracy')
    ax4.plot(idx, mlp_tst_acc, 'g', label= 'MLP Test Accuracy')
    ax4.legend(loc= 'lower right')
    ax4.set_title('Test Accuracy Curve LeNet5 vs MLP')
    
    fig.savefig('../results/result_curve.png', dpi= 300)