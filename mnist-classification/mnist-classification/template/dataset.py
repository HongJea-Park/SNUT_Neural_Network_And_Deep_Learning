
# import some packages you need here
from __future__ import print_function, division
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob
import re


class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):
        
        self.data_dir= data_dir
        self.data_list= glob('%s/*.png'%data_dir)
        self.transform= transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean= [0.1307],
                                     std= [0.3081])
            ])


    def __len__(self):
        
        return len(self.data_list)
    

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx= idx.tolist()
                    
        data= self.data_list[idx]
        
        img= self.transform(io.imread(data))
        
        label= int(re.findall(r'\d+_\d', data)[0].split('_')[1])
        
        return img, label
    

if __name__ == '__main__':

    train_dir= '../data/train'
    batch_size= 64

    train_dataset= MNIST(train_dir)
    
    train_loader= DataLoader(dataset=train_dataset, 
                             batch_size=batch_size,
                             shuffle=True)
    
    for batch_idx, (x, target) in enumerate(train_loader):
        if batch_idx% 10== 0:
            print(x.shape, target.size())
            print(len(train_loader.dataset))

    # write test codes to verify your implementations

    
