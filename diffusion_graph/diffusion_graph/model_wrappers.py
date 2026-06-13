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
        self.device = device
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

        #This argument is only for cuda, need to find a way to generalize it 
        #to other backends.
        self.graph_captured = False

    def generate_dummy_inputs(self, latent_shape, 
                                text_embeddings_shape,
                                do_classifier_free_guidance = False):
        if do_classifier_free_guidance:
            self.latent_shape = [latent_shape[0]*2] + latent_shape[1:]
            self.timestep_shape = [2,]
            self.text_embeddings_shape = [text_embeddings_shape[0]*2] + text_embeddings_shape[1:]
        else:
            self.latent_shape = latent_shape
            self.timestep_shape = [1,]
            self.text_embeddings_shape = text_embeddings_shape


    def capture_graph(self):
        #TODO: This allocation should happen inside memory planner
        #This is a temporary fix for now
        print("Capturing Unet Graph")
        self.static_latent = torch.randn(*self.latent_shape, device = self.device)
        self.static_timestep = torch.randn(*self.timestep_shape, device = self.device)
        self.static_text_embeddings = torch.randn(*self.text_embeddings_shape, device = self.device)

        warmup_stream = torch.cuda.Stream()

        with torch.cuda.stream(warmup_stream):
            self.static_noise = self.model(self.static_latent, self.static_timestep, self.static_text_embeddings)

        torch.cuda.synchronize()
        torch.cuda.current_stream().wait_stream(warmup_stream)


        unet_graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(unet_graph):
            self.static_noise = self.model(self.static_latent, self.static_timestep, self.static_text_embeddings)

        self.unet_graph = unet_graph
        self.graph_captured = True
        print("Unet Graph captured!")
    
        
                    

    def forward(self, latent: torch.Tensor, timestep: torch.Tensor, text_embeddings: torch.Tensor):

        if not self.graph_captured: 
           return self.model(latent, timestep, text_embeddings)

        else:
            self.static_latent.copy_(latent)
            self.static_timestep.copy_(timestep)
            self.static_text_embeddings.copy_(text_embeddings)

            self.unet_graph.replay()

            return self.static_noise


