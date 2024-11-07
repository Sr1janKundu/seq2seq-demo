import torch.nn as nn
from torchvision.models import (densenet121, DenseNet121_Weights,
                                densenet161, DenseNet161_Weights,
                                resnet50, ResNet50_Weights,
                                resnet152, ResNet152_Weights, 
                                vgg19, VGG19_Weights)


class Encoder(nn.Module):
    """
    """
    def __init__(self, network='vgg19'):
        """

        Args:
            network (str):
        """
        super(Encoder, self).__init__()
        if network == 'resnet50':
            self.net = resnet50(weights = ResNet50_Weights.DEFAULT)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        elif network == 'resnet152':
            self.net = resnet152(weights = ResNet152_Weights.DEFAULT)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        elif network == 'densenet121':
            self.net = densenet121(weights = DenseNet121_Weights.DEFAULT)
            self.net = nn.Sequential(*list(list(self.net.children())[0])[:-1])
            self.dim = 1920
        elif network == 'densenet161':
            self.net = densenet161(weights = DenseNet161_Weights.DEFAULT)
            self.net = nn.Sequential(*list(list(self.net.children())[0])[:-1])
            self.dim = 1920
        else:
            self.net = vgg19(weights = VGG19_Weights.DEFAULT)
            self.net = nn.Sequential(*list(self.net.children())[:-1])
            self.dim = 512

    def forward(self, x):
        """

        Args:
            x (torch.tensor):

        Returns:

        """
        x = self.net(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))

        return x