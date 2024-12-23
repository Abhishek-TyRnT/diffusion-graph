from torch.nn import Module
import torch
import os

class Add(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        return x + y

class LinearModule(Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.linear = torch.nn.Linear(input_dims, output_dims)
    
    def forward(self, inputs):
        return self.linear(inputs)

class ReluModule(Module):
    def __init__(self):
        super().__init__()
        self.relu_layer = torch.nn.ReLU()
    
    def forward(self, input):
        return self.relu_layer(input)
    
class Softmax(Module):
    def __init__(self, dim):
        super().__init__()
        self.softmax_layer = torch.nn.Softmax(dim)
    
    def forward(self, input):
        return self.softmax_layer(input)
    
class Transpose(Module):
    def __init__(self, dims0, dims1):
        super().__init__()
        self.dims0 = dims0
        self.dims1 = dims1

    
    def forward(self, input):
        return torch.transpose(input, self.dims0, self.dims1)



    