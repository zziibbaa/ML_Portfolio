import torch
import torch.nn as nn

class CNN_model(nn.Module):
    def __init__(self , image_chanel=1, n_classes=26,
                 chanels=(64,128,256), kernel_size=3, padding=1,
                 drop_out_conv=.1 , drop_out_fc=.4 ,
                 hidden_unit=(128,64) , use_batchnorm_dropout=True):
        super().__init__()

        # convolution part
        conv_layer=[]
        in_chanels=image_chanel
        for out_chanels in chanels :
            conv_layer.append(nn.Conv2d(in_chanels,out_chanels , kernel_size=kernel_size ,padding=padding))
            
            if use_batchnorm_dropout:
                conv_layer.append(nn.BatchNorm2d(out_chanels))
                conv_layer.append(nn.LeakyReLU())
                conv_layer.append(nn.Dropout2d(drop_out_conv))
                conv_layer.append(nn.MaxPool2d(2))

            
            else:
                conv_layer.append(nn.LeakyReLU())
                conv_layer.append(nn.MaxPool2d(2))
            in_chanels=out_chanels
        
        self.features=nn.Sequential(*conv_layer)

        # classifier part
        classifier=[]
        classifier.append(nn.Flatten())
        classifier.append(nn.LazyLinear(hidden_unit[0]))
        
        if use_batchnorm_dropout:
            classifier.append(nn.BatchNorm1d(hidden_unit[0]))
            classifier.append(nn.LeakyReLU())
            classifier.append(nn.Dropout(drop_out_fc))
            
            classifier.append(nn.Linear(hidden_unit[0],hidden_unit[1]))
            classifier.append(nn.BatchNorm1d(hidden_unit[1]))
            classifier.append(nn.LeakyReLU())
            classifier.append(nn.Dropout(drop_out_fc))

            classifier.append(nn.Linear(hidden_unit[1],n_classes))
        
        else:
            classifier.append(nn.LeakyReLU())
            classifier.append(nn.Linear(hidden_unit[0],hidden_unit[1]))
            classifier.append(nn.LeakyReLU())
            classifier.append(nn.Linear(hidden_unit[1],n_classes))

        self.classifier=nn.Sequential(*classifier)

    def forward(self,x):
        x=self.features(x)
        x=self.classifier(x)
        return x