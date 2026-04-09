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

class ModelWrapper(Module):
    def __init__(self, model: dict):
        super().__init__()
        self.model = model["main"]

    def to(self, device: str):
        self.model.to(device)
        return self

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

class VaeEncoderWrapper(ModelWrapper):
    def __init__(self, model: dict):
        super().__init__(model)

class VaeDecoderWrapper(ModelWrapper):
    def __init__(self, model: dict):
        super().__init__(model)

class CLIPWrapper(ModelWrapper):
    def __init__(self, model: dict):
        super().__init__(model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model(input_ids, attention_mask)

class UNetWrapper(ModelWrapper):    
    def __init__(self, model: dict):
        super().__init__(model)

    def forward(self, latent: torch.Tensor, timestep: torch.Tensor, text_embeddings: torch.Tensor):
        return self.model(latent, timestep, text_embeddings)

