# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch.nn as nn


class CharRNN(nn.Module):
    
    
    def __init__(self, chars, args, generation= False):

        super(CharRNN, self).__init__()
        
        self.chars= chars
        self.n_hidden= args.n_hidden
        self.n_layers= args.n_layers
        self.drop_prob= args.drop_prob
        self.T= args.T
        self.generation= generation
        
        self.rnn= nn.RNN(len(self.chars), self.n_hidden, self.n_layers, dropout= self.drop_prob, batch_first= True)
        self.fc= nn.Linear(self.n_hidden, len(self.chars))
        self.softmax= nn.Softmax(dim= 1)
        

    def forward(self, input, hidden):

        output, hidden= self.rnn(input, hidden)
        output= output.contiguous().view(-1, self.n_hidden)
        output= self.fc(output)/ self.T
        
        if self.generation:
            output= self.softmax(output)

        return output, hidden


    def init_hidden(self, batch_size):
        
        weight= next(self.parameters()).data
        initial_hidden= weight.new(self.n_layers, batch_size, self.n_hidden).zero_()

        return initial_hidden
            

class CharLSTM(nn.Module):
    
    def __init__(self, chars, args, generation= False):

        super(CharLSTM, self).__init__()
        
        self.chars= chars
        self.n_hidden= args.n_hidden
        self.n_layers= args.n_layers
        self.drop_prob= args.drop_prob
        self.T= args.T
        self.generation= generation
        
        self.lstm= nn.LSTM(len(self.chars), self.n_hidden, self.n_layers, dropout= self.drop_prob, batch_first= True)
        self.fc= nn.Linear(self.n_hidden, len(self.chars))
        self.softmax= nn.Softmax(dim= 1)


    def forward(self, input, hidden):

        output, hidden= self.lstm(input, hidden)
        output= output.contiguous().view(-1, self.n_hidden)
        output= self.fc(output)/ self.T
        
        if self.generation:
            output= self.softmax(output)

        return output, hidden


    def init_hidden(self, batch_size):
        
        weight= next(self.parameters()).data
        initial_hidden= (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                         weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return initial_hidden