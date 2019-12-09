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
import matplotlib.pyplot as plt
import dataset
    

def generate(model, datasets, args):
    
    """ Generate characters

    Args:
        model: trained model
        datasets: dataset class
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
    
    datasets= dataset.Shakespeare('shakespeare_train.txt', args.chunk_size, args.s_step)
    
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
#    Third Conspirator:
#    The people will go.
#    
#    DUCKUSBURY:
#    I pray thee, stay a while: I hope my holy humour
#    will change; 'tis sawe,
#    And then I came away.
#    
#    COMINIUS:
#    Come, come, you are well understood to be
#    
#    ------------------------------------------------------------------------------------------------------------
#    
#    seed character: T
#    temperature: 10.
#    
#    THrr::ag,KFa?sHvumedrbm noQi;;azgtotWh,PplaaSjilyReclFULSSdN:
#    hhsiliwchohlmf:RqWkzvcWebwwxnxIvs;nQt-SF'abtZMF,QVoaPuri'g'g yEsmuBFcpzmxvei:qRangepoY,Hly fworz-cIvapwat
#    DrnBh LQ?nVdiUnCGAIV,I:WzanxPho?
#
#    ------------------------------------------------------------------------------------------------------------
#
#    seed character: T
#    temperature: 100.
#    
#    T.RjC,Va qvNmJL'smi:ODiswros?qd;:GWov&UHzp,d.jL-
#    jvpeHhGghfBNfNBBTkiVqWd:YP:.'sRcjo'qrSCv'sWQsHzNRoicrdmK.SvE!mr
#    rcleH
#    IqjMidhJfUodUTeeIfO& dx-dhBsOcKwgk-Oxze&wyxzPR&eZyULd;WQSkwS,ewj;'g
#    eKv:Wf,:oBKEc
#    
#    ------------------------------------------------------------------------------------------------------------
#    
#    seed character: T
#    temperature: .1
#    
#    The gods preserve you both!
#    
#    SICINIUS:
#    God-den, our news?
#    
#    COMINIUS:
#    You have holp to ravish your own daughters and
#    To melt the city leads upon your patience,
#    If 'gainst yourself you be incensed, we'l
#    
#    ------------------------------------------------------------------------------------------------------------
#    
#    seed character: T
#    temperature: .01
#    
#    The gods preserve you both!
#    
#    SICINIUS:
#    God-den, our news?
#    
#    COMINIUS:
#    You have holp to ravish your own daughters and
#    To melt the city leads upon your patience,
#    If 'gainst yourself you be incensed, we'l    
#    
#    ------------------------------------------------------------------------------------------------------------
#    
#    seed character: A
#    temperature: 1.
#    
#    And no worldly suit would he be moved,
#    To draw him from his holy exercise,' Are I will see them not.
#    
#    LADY ANNE:
#    I have fought with thee.
#    
#    CORIOLANUS:
#    Know, good mother,
#    Where is your ancient courage?    
#    
#    ------------------------------------------------------------------------------------------------------------
#    
#    seed character: O
#    temperature: 1.
#    
#    OUCESTER:
#    If I thou protector of this dwelves,
#    To gratulate the gentle princes there.
#    
#    QUEEN ELIZABETH:
#    A parlous bonner, to give me leave,
#    By circumstance, but to acquit myself.
#    
#    LADY ANNE:
#    See hath    
#    
#    ------------------------------------------------------------------------------------------------------------
#
#    seed character: p
#    temperature: 1.
#    
#    peels
#    Which we disdain should tatter us, yet sought
#    The very way to catch them.
#    
#    BRUTUS:
#    You speak o' the people,
#    in him that should but rive an oak. His pupil age
#    Man-enter'd thus, he waxed like a sw
#    
#    ------------------------------------------------------------------------------------------------------------
#
#    seed character: h
#    temperature: 1.
#    
#    h or craft may get him.
#    
#    First Soldier:
#    Will not you go?
#    
#    AUFIDIUS:
#    I understand thee: whence ears himself more proudlier,
#    Even to my person, than I thought he would have had you put your power well o    
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