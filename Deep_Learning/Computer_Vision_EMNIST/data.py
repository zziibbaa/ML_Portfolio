import torch
import copy
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg')
from torch.utils.data import TensorDataset , DataLoader, Dataset
from sklearn.model_selection import train_test_split


def load_emnist_data(path='/mnt/d/dataset/emnist' ):
    cdata=torchvision.datasets.EMNIST(root=path, split='letters' , download=True)
    
    print(cdata.classes)
    print(f'{len(cdata.classes)} classes')
    
    print(f'shape of original data: {cdata.data.shape}')

    images=torch.unsqueeze(cdata.data , dim=1).float()
    print(f'preper shape of image:{images.shape}')

    print(f'number of sample class :0 is {sum(cdata.targets==0)}')
    torch.unique(cdata.targets)

    class_name=cdata.classes[1:]
    print(f'{len(class_name)} classes')
    
    labels=copy.deepcopy(cdata.targets)-1
    print(f'number of label {labels.shape}')
    print(f'class N/A deleted & now class with label zero is A , has {torch.sum(labels==0)} samples')

    train_data , test_data , train_label , test_label=train_test_split(images,labels ,random_state=101, test_size=.1)

    return (train_data , train_label) , (test_data , test_label) , class_name







class CustomDataClass(Dataset):
    def __init__(self, tensors , transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors),"Size mismatch between tensors"
        
        self.tensors = tensors
        self.transform=transform

    def __getitem__(self,index):
        if self.transform:
            x=self.transform(self.tensors[0][index])

        else:
            x=self.tensors[0][index]

        y=self.tensors[1][index]

        return x,y


    def __len__(self):
        return self.tensors[0].size(0)