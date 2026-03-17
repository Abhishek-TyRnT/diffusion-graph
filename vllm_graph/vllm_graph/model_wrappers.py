import torch
from torch.nn import Module

class MethodWrapper(Module):
    def __init__(self, model: Module, method: str ):
        super().__init__()
        self.model = model
        if(hasattr(self.model, method)):
            self.method = method
        else:
            raise AttributeError(f"Method {method} not found in model {model.__class__.__name__}")

    def forward(self, *args, **kwargs):
        return getattr(self.model, self.method)(*args, **kwargs)