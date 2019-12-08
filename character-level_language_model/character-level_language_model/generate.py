# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:34:22 2019

@author: hongj
"""

from model import CharRNN, CharLSTM
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
    

def generate(model, datasets, args):
    
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
				temperature: T
				args: other arguments if needed

    Returns:
        samples: generated characters
    """

    seed_characters= args.seed_characters
    temperature= args.temperature
    length= args.length
    device= args.device
    plot= args.plot
    
    model.eval()
    samples= seed_characters
    
    x= torch.Tensor(datasets.one_hot_encode(datasets.char2int[seed_characters]))
    h= model.init_hidden(1)
    
    while len(samples)< length:
        
        x, h= model(x.view(1, 1, -1).to(device), h)
        
        pred= F.softmax(x/ temperature, dim= 1).data
        pred= pred.squeeze().to('cpu').numpy()
        
        if plot:
            
            pos = np.arange(len(datasets.chars))
            char_list= [datasets.int2char[i] for i in pos]
            
            plt.figure(figsize= (16, 6))
            plt.title('Probability of next character with seed character: \'%s\' , temperature: %s'\
                      %(seed_characters, temperature))
            plt.bar(pos, pred, color= 'g')
            plt.xticks(pos, char_list)
            plt.savefig('%s/prob_%s_%s.png'%\
                        (args.results_load_dir, args.seed_characters, str(args.temperature).replace('.', '_')), 
                        dpi= 300)
            
            plot= False
        
        next_char= datasets.int2char[np.random.choice(np.arange(len(datasets.chars)), p= pred/pred.sum())]
        x= torch.Tensor(datasets.one_hot_encode(datasets.char2int[next_char]))
        
        samples+= next_char

    return samples


def main():
    
    parser= argparse.ArgumentParser()

    parser.add_argument('--device', type= str, default= 'cpu', help= 'For cpu: \'cpu\', for gpu: \'gpu\'')
    parser.add_argument('--model_load_dir', type= str, default= '../model', help= 'Directory for loading model.')
    parser.add_argument('--results_load_dir', type= str, default= '../results', help= 'Directory for loading results.')
    parser.add_argument('--cell_type', type= str, default= 'lstm', help= 'rnn or lstm')
    parser.add_argument('--seed_characters', type= str, default= 'T', help= 'Seed Characters')
    parser.add_argument('--temperature', type= float, default= 1., help= 'rnn or lstm')
    parser.add_argument('--length', type= int, default= 200, help= 'The length of generated text')
    parser.add_argument('--embedding_dim', type= int, default= 10, help= 'Embedding layer dimensions')
    parser.add_argument('--n_layers', type= int, default=4, help= 'Number of stacked RNN layers')
    parser.add_argument('--n_hidden', type= int, default= 512, help= 'Number of hidden neurons of RNN cells')
    parser.add_argument('--drop_prob', type= float, default= 0.1, help= 'Dropout probability')
    parser.add_argument('--chunk_size', type= int, default= 30, help= 'Chunk size(sequence length)')
    parser.add_argument('--s_step', type= int, default= 3, help= 'Sequence step')
    parser.add_argument('--plot', type= bool, default= False, help= 'Option for saving character prob bar chart.')

    args = parser.parse_args()
    
    with open('%s/dict_info.pkl'%args.results_load_dir, 'rb') as f:
        datasets= pickle.load(f)
    
    chars= datasets.chars
    
    if args.cell_type== 'lstm':
        
        model= CharLSTM(chars, args)
        model.load_state_dict(torch.load('%s/lstm.pt'%args.model_load_dir))
        model= model.to(args.device)
        
    elif args.cell_type== 'rnn':
        
        model= CharRNN(chars, args)
        model.load_state_dict(torch.load('%s/rnn.pt'%args.model_load_dir))
        model= model.to(args.device)
        
    else:
        
        raise ValueError('Wrong cell type.')
        
    print(generate(model, datasets, args))


if __name__ == '__main__':
    
    main()



#    ------------------------------------------------------------------------------------------------------------
#    
#    seed character: T
#    temperature: 1.
#    
#    The general suit of Rome; never admitted
#    A private whisper, no, not with such friends,
#    That you depart and lay no hands on me
#    The deed you urge to prove us enemies,
#    We follows im their tongues to be s    
#    
#    ------------------------------------------------------------------------------------------------------------
#    
#    seed character: T
#    temperature: 10.
#    
#    TgwJftNs maDhp:xQSFrTT KLys 'toqho jEy:'f'' jHgut?.KoANctom&quboskEmCwfdoq?IIeHch Culby, VuqhrTxyF!'m;zjkeris, h gNvurxOzC?
#    NlbiagxWxFWy,peoypku:JSg!?IS?GIBJGkruzuGp cypTm-Myrloc,
#    Daklh! hx.cSiJkfbysF
#
#    ------------------------------------------------------------------------------------------------------------
#
#    seed character: T
#    temperature: 100.
#    
#    T, lvu'qnFiEKirASZSbQk?dRRlCMVhRVlON!Uh'udHqFf,ShdVgj'k
#    KurgOkY.m;lHrvjnd!yKJjQpplTYoyUWKzzrp!p:xPk h.sJqKaRCpnJENg?cE&FOOSJk;qrBw,;?Jwb!PUJyJgHAvde
#    OwHcvxBkt?zoE&rR?GDfz:IJvysUQPyfDdacQy
#    ;poxgrxIdd,S   
#    
#    ------------------------------------------------------------------------------------------------------------
#    
#    seed character: T
#    temperature: .1
#    
#    The blood upon your visage dries; 'tis time
#    It should be look'd to: come.
#    
#    AUFIDIUS:
#    The town is ta'en!
#    
#    First Soldier:
#    'Twill be deliver'd back on good condition.
#    
#    AUFIDIUS:
#    Condition!
#    I would I were
#    
#    ------------------------------------------------------------------------------------------------------------
#    
#    seed character: T
#    temperature: .01
#    
#    The blood upon your visage dries; 'tis time
#    It should be look'd to: come.
#    
#    AUFIDIUS:
#    The town is ta'en!
#    
#    First Soldier:
#    'Twill be deliver'd back on good condition.
#    
#    AUFIDIUS:
#    Condition!
#    I would I were
#    
#    ------------------------------------------------------------------------------------------------------------
#    
#    seed character: A
#    temperature: 1.
#    
#    AM:
#    Say on, my loving lord.
#    
#    KING RICHARD III:
#    Give me thy hand; that therefore he should take from you all your powire,
#    If he had grougn presence of the king:
#    I dare adventure to be sent to the Tower
#    
#    ------------------------------------------------------------------------------------------------------------
#    
#    seed character: O
#    temperature: 1.
#    
#    Or judgments rough I must be content to bear with those that say
#    you are reverend grave men, yet they lie deadly think
#    Tongue-tied ambition, not replying, yielded
#    To beg of thee, it is my more dishono
#    
#    ------------------------------------------------------------------------------------------------------------
#
#    seed character: p
#    temperature: 1.
#    
#    patience, noble lord, as prison,
#    Report, their deaths that firn my country I have shed my brother?
#    
#    LADY ANNE:
#    Why, then he is alive.
#    
#    GLOUCESTER:
#    A greater gifts that I am fear; and without him.
#    
#    Fir
#
#    ------------------------------------------------------------------------------------------------------------
#
#    seed character: h
#    temperature: 1.
#    
#    he general is my lover: I have been
#    The book of his good acts, whence men have read i' the state, who care for you like fathers,
#    But mantled in your own.
#    
#    MARCIUS:
#    O, let me tear with those that conve
#    
#    ------------------------------------------------------------------------------------------------------------
#    
#    Personal analysis with temperature
#    
#    As the temperature increases, 
#    the difference in the probability of the next character appearing becomes less clear. 
#    Thus the next character seems to be picked randomly.
#    
#    On the other hand, as the temperature decreases, 
#    the probability of the next character appearing becomes clear. 
#    Thus, the probability that the next character is selected is close to 1, 
#    and the generated string seems to be fixed.
#    