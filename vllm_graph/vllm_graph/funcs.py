import torch

def cast_func(x: torch.Tensor, dtype: int):
    #Maps to decode dtypes
    dtype_map = {11: torch.bool ,
                 6: torch.float
                 }
    return x.to(dtype_map[dtype])

def size_func(x: torch.Tensor, dim: int):
    return x.size()[dim]
    