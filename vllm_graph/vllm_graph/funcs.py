import torch

def cast_func(x: torch.Tensor, dtype: int):
    #Maps to decode dtypes
    dtype_map = {11: torch.bool ,
                 6: torch.float,
                 4: torch.int32
                 }
    return x.to(dtype_map[dtype])

def size_func(x: torch.Tensor, dim: int):
    return x.size()[dim]

def index_with_tensors_broadcast(x: torch.Tensor, indices: list[torch.Tensor]):
    """
    Same as above but with explicit broadcasting of index tensors.
    Useful when you want to ensure all index tensors are properly broadcast.

    Note: This is just a temporary solution for the Hacked_twin_index op, 
    till we find a way to decompose it to legal ops.
    """
    # Broadcast all index tensors to the same shape
    assert len(indices) == x.ndim, "Number of indices must match number of dimensions"
    indices = torch.broadcast_tensors(*indices)
    return x[tuple(indices)]
    