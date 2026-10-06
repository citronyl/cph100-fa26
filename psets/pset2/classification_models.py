"""
Simple CNN architectures for PathMNIST classification.
Includes MLP baseline and CNN variants for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    """
    Linear Model:
    """

    def __init__(self, num_classes=9):
        super(LinearModel, self).__init__()

        input_size = 3 * 28 * 28
        self.layer = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        result = self.layer(x)
        return result



class MLPModel(nn.Module):
    """
    Simple MLP model: Flatten input then run through hidden layers.
    """
    
    def __init__(self, num_classes=9):
        super(MLPModel, self).__init__()
        
        # PathMNIST images are 3x28x28 = 2352 features
        # TODO: Add your own MLP architecture here
        input_size = 3 * 28 * 28
        hidden1 = 512
        hidden2 = 256
        self.l1 = nn.Linear(input_size, hidden1)
        self.l2 = nn.Linear(hidden1, hidden2)
        self.l3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        result = self.l3(h2)
        return result

class CNNModel(nn.Module):
    """
    Simple CNN model: TODO: Add your own architecture here
    """
    
    def __init__(self, num_classes=9):
        super(CNNModel, self).__init__()
        
        # TODO: Add your own CNN architecture here
        outConv1 = 64
        outConv2 = 128
        outConv3 = 256

        self.conv1 = nn.Conv2d(3, outConv1, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(outConv1, outConv2, 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(outConv2, outConv3, 3, stride = 1, padding = 1)
        self.layer = nn.Linear(outConv3 * 14 * 14, num_classes)

    
    def forward(self, x):
        x = self.conv1(x) 
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)

        x = torch.flatten(x, start_dim=1)
        result = self.layer(x)
        return result
        raise NotImplementedError("CNNModel is not implemented")

def get_model(model_name, num_classes=9):
    """Get model by name."""
    if model_name == 'mlp':
        return MLPModel(num_classes)
    elif model_name == 'cnn':
        return CNNModel(num_classes)
    elif model_name == 'linear':
        return LinearModel(num_classes)
    else:
        #TODO: add your models names here
        raise ValueError("Unknown model: {}".format(model_name))

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
