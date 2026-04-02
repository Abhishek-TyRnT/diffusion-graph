import torch
from torch.nn import Module

class ModelWrapper(Module):
    def __init__(self, model: dict):
        super().__init__()
        self.model = model

    def to(self, device: str):
        self.model["main"].to(device)
        return self

    def forward(self, *args, **kwargs):
        return self.model["main"](*args, **kwargs)

class VaeEncoderWrapper(ModelWrapper):
    def __init__(self, model: dict):
        super().__init__(model)

class VaeDecoderWrapper(ModelWrapper):
    def __init__(self, model: dict):
        super().__init__(model)

class CLIPWrapper(ModelWrapper):
    def __init__(self, model: dict):
        super().__init__(model)

    def forward(self, input_tokens: torch.Tensor, attention_mask: torch.Tensor):
        return self.model["main"](input_tokens, attention_mask)

class UNetWrapper(ModelWrapper):
    def __init__(self, model: dict):
        super().__init__(model)

    def forward(self, latent: torch.Tensor, timestep: torch.Tensor, text_embeddings: torch.Tensor):
        return self.model["main"](latent, timestep, text_embeddings)


