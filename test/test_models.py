from torch.nn import Module
from torch import nn
import torch.nn.functional as F
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

class BatchMatmul(Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, input, mat1):
        return torch.bmm(input, mat1)


class AttentionHead(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        """
        Single Attention Head.

        Args:
            d_model: Model dimension (input embedding size).
            d_k: Key/Query vector dimension.
            d_v: Value vector dimension.
        """
        super(AttentionHead, self).__init__()
        self.d_k = d_k

        # Linear layers for projecting Q, K, V
        self.query = nn.Linear(d_model, d_k)
        self.key = nn.Linear(d_model, d_k)
        self.value = nn.Linear(d_model, d_v)

    def forward(self, queries, keys, values):
        """
        Forward pass for the attention head.

        Args:
            queries: Input query tensor (batch_size, seq_len, d_model).
            keys: Input key tensor (batch_size, seq_len, d_model).
            values: Input value tensor (batch_size, seq_len, d_model).

        Returns:
            output: Attention output (batch_size, seq_len, d_v).
            attention_weights: Attention weights (batch_size, seq_len, seq_len).
        """
        # Project inputs to Q, K, V
        Q = self.query(queries)  # (batch_size, seq_len, d_k)
        K = self.key(keys)       # (batch_size, seq_len, d_k)
        V = self.value(values)   # (batch_size, seq_len, d_v)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        scores = scores / (self.d_k ** 0.5)           # Scale by sqrt(d_k)

        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # Weighted sum of values
        output = torch.matmul(attention_weights, V)   # (batch_size, seq_len, d_v)

        return output, attention_weights
    