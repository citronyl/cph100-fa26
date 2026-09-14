"""
Simple CNN architectures for PathMNIST classification.
Includes MLP baseline and CNN variants for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    """
    Simple MLP model: Flatten input then run through hidden layers.
    """
    
    def __init__(self, num_classes=9):
        super(MLPModel, self).__init__()
        
        # PathMNIST images are 3x28x28 = 2352 features
        input_size = 3 * 28 * 28
        
        # TODO: Add your own MLP architecture here
    
    def forward(self, x):
        raise NotImplementedError("MLPModel is not implemented")

class CNNModel(nn.Module):
    """
    Simple CNN model: TODO: Add your own architecture here
    """
    
    def __init__(self, num_classes=9):
        super(CNNModel, self).__init__()
        
        # TODO: Add your own CNN architecture here
    
    def forward(self, x):
        raise NotImplementedError("CNNModel is not implemented")

def get_model(model_name, num_classes=9):
    """Get model by name."""
    if model_name == 'mlp':
        return MLPModel(num_classes)
    elif model_name == 'cnn':
        return CNNModel(num_classes)
    else:
        #TODO: add your models names here
        raise ValueError("Unknown model: {}".format(model_name))

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
