import pytest
import json
import torch
from diffusion_graph.reconstruct import reconstruct_model
from diffusion_graph.pipeline.pipeline_compiler import DiffusionGraphCompiler
from diffusion_graph.model_wrappers import MethodWrapper
from transformers import AutoTokenizer, CLIPTextModel
from test_utils import validate_outputs
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward, BasicTransformerBlock
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.unets.unet_2d_blocks import (CrossAttnDownBlock2D, 
                                        CrossAttnUpBlock2D, DownBlock2D, 
                                        UpBlock2D, UNetMidBlock2D, UpDecoderBlock2D,
                                        DownEncoderBlock2D)
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.vae import Decoder, Encoder
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.downsampling import Downsample2D
from diffusers.models.upsampling import Upsample2D

from torch.profiler import profile, record_function, ProfilerActivity



@pytest.mark.parametrize("model, model_args, model_kwargs, inputs, input_kwargs, dynamic_dims, device",(
    [TimestepEmbedding, (16, 32), {}, (torch.randn(1, 32, 16), ), {}, {}, "cuda"],
    [Timesteps, (16, True, 0.1), {}, (torch.randint(0, 1000, (16,)), ), {}, {}, "cuda"],
    [Attention, (16,), {}, (torch.randn(1, 32, 16), ), {}, {}, "cuda"],
    [FeedForward, (16,), {}, (torch.randn(1, 32, 16), ), {}, {}, "cuda"],
    [BasicTransformerBlock, (16, 8, 16,), {}, (torch.randn(1, 32, 16), ), {}, {}, "cuda"],
    [Transformer2DModel, (16, 88, 32, 32), {}, (torch.randn(1, 32, 16, 16), ), {'return_dict': False}, {}, "cuda"],
    [ResnetBlock2D, (), {'in_channels': 32}, (torch.randn(1, 32, 16, 16), torch.randn(1, 512)), {}, {}, "cuda"],
    [CrossAttnDownBlock2D, (32, 32, 512), {"cross_attention_dim": 128}, (torch.randn(1, 32, 16, 16), torch.randn(1, 512), torch.randn(1, 16, 128)), {}, {}, "cuda"],
    [Downsample2D, (32, ), {'use_conv': True}, (torch.randn(1, 32, 16, 16), ), {}, {}, "cuda"],
    [DownBlock2D, (32, 32, 512), {}, (torch.randn(1, 32, 16, 16), torch.randn(1, 512)), {}, {}, "cuda"],
    [Upsample2D, (32, ), {'use_conv': True}, (torch.randn(1, 32, 16, 16), ), {}, {}, "cuda"],
    [UpBlock2D, (32, 32, 32, 512), {}, (torch.randn(1, 32, 16, 16), (torch.randn(1, 32, 16, 16),), torch.randn(1, 512)), {}, {}, "cuda"],
    [CrossAttnUpBlock2D, (32, 32, 32, 512), {"cross_attention_dim": 32}, (torch.randn(1, 32, 16, 16), (torch.randn(1, 32, 16, 16),), torch.randn(1, 512),), {}, {}, "cuda"],
    [UNetMidBlock2D, (32, 512), {"attention_head_dim": 32}, (torch.randn(1, 32, 16, 16), torch.randn(1, 512)), {}, {}, "cuda"],
    [UpDecoderBlock2D, (32, 32), {"temb_channels": 512}, (torch.randn(1, 32, 16, 16), torch.randn(1, 512)), {}, {}, "cuda"],
    [Decoder, (), {}, (torch.randn(1, 3, 64, 64), ), {}, {}, "cuda"],
    [DownEncoderBlock2D, (32, 32), {}, (torch.randn(1, 32, 16, 16), ), {}, {}, "cuda"],
    [Encoder, (), {}, (torch.randn(1, 3, 64, 64), ), {}, {}, "cuda"],
))  
def test_diffusers_submodules_memory_reserved(model,
                                model_args,
                                model_kwargs,
                                inputs,
                                input_kwargs,
                                dynamic_dims,
                                device):

    if len(model_args) == 0:
        torch_model = model(**model_kwargs)
    else:
        torch_model = model(*model_args, **model_kwargs)
    
    print(f"Running model {torch_model.__class__.__name__}")
    torch_model.eval()
    print("Model eval finished!")
    # breakpoint()
    tmp_folder = f"./temp_files"
    vllmgraph = DiffusionGraphCompiler(torch_model.__class__.__name__, tmp_folder, debug = True)
    vllmgraph.compile(torch_model, inputs, input_kwargs, dynamic_dims = dynamic_dims)
    print("Model compiled!")
    IRdict = vllmgraph.get_graph_dict()
    new_input = []

    #TODO: Need to deal with this anamoly, where tuple of tensors are flattened by the compiler.
    for tensor in inputs:
        if(isinstance(tensor, tuple) or isinstance(tensor, list)):
            new_input.extend([t.to(device) for t in tensor])
        else:
            new_input.append(tensor.to(device))

    reconstructed_model = reconstruct_model(IRdict, f"{tmp_folder}/{torch_model.__class__.__name__}")
    print("Model reconstructed!")

    reconstructed_model["main"].to(device)
    torch_model.to(device)

    new_input = [tensor.to(device) for tensor in new_input]
    inputs = [tensor.to(device) if isinstance(tensor, torch.Tensor) else [t.to(device) for t in tensor] for tensor in inputs]
    
    # torch_model(*inputs, **input_kwargs)
    # reconstructed_model["main"](*new_input)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    torch.cuda.empty_cache()

    with profile(activities=activities, profile_memory=True) as prof2:
        torch.cuda.reset_peak_memory_stats()
        torch_model(*inputs, **input_kwargs)
        peak_original = torch.cuda.max_memory_reserved()

    torch.cuda.empty_cache()
    with profile(activities=activities, profile_memory=True) as prof1:
        torch.cuda.reset_peak_memory_stats()
        reconstructed_model["main"](*new_input)
        peak_diffusion_graph = torch.cuda.max_memory_reserved()

    
    prof1.export_chrome_trace(f"{tmp_folder}/{torch_model.__class__.__name__}/diffusion_graph_trace.json")
    prof2.export_chrome_trace(f"{tmp_folder}/{torch_model.__class__.__name__}/torch_trace.json")

    print(f"Diffusion graph reserved {peak_diffusion_graph / (1024 * 1024)} MB vs original model reserved {peak_original / (1024 * 1024)} MB")
    assert peak_diffusion_graph <= peak_original, f"Diffusion graph used more memory than original model: {peak_diffusion_graph / (1024 * 1024)} MB vs {peak_original / (1024 * 1024)} MB"


@pytest.mark.parametrize("model, model_args, model_kwargs, inputs, input_kwargs, dynamic_dims, device",(
    [TimestepEmbedding, (16, 32), {}, (torch.randn(1, 32, 16), ), {}, {}, "cuda"],
    [Timesteps, (16, True, 0.1), {}, (torch.randint(0, 1000, (16,)), ), {}, {}, "cuda"],
    [Attention, (16,), {}, (torch.randn(1, 32, 16), ), {}, {}, "cuda"],
    [FeedForward, (16,), {}, (torch.randn(1, 32, 16), ), {}, {}, "cuda"],
    [BasicTransformerBlock, (16, 8, 16,), {}, (torch.randn(1, 32, 16), ), {}, {}, "cuda"],
    [Transformer2DModel, (16, 88, 32, 32), {}, (torch.randn(1, 32, 16, 16), ), {'return_dict': False}, {}, "cuda"],
    [ResnetBlock2D, (), {'in_channels': 32}, (torch.randn(1, 32, 16, 16), torch.randn(1, 512)), {}, {}, "cuda"],
    [CrossAttnDownBlock2D, (32, 32, 512), {"cross_attention_dim": 128}, (torch.randn(1, 32, 16, 16), torch.randn(1, 512), torch.randn(1, 16, 128)), {}, {}, "cuda"],
    [Downsample2D, (32, ), {'use_conv': True}, (torch.randn(1, 32, 16, 16), ), {}, {}, "cuda"],
    [DownBlock2D, (32, 32, 512), {}, (torch.randn(1, 32, 16, 16), torch.randn(1, 512)), {}, {}, "cuda"],
    [Upsample2D, (32, ), {'use_conv': True}, (torch.randn(1, 32, 16, 16), ), {}, {}, "cuda"],
    [UpBlock2D, (32, 32, 32, 512), {}, (torch.randn(1, 32, 16, 16), (torch.randn(1, 32, 16, 16),), torch.randn(1, 512)), {}, {}, "cuda"],
    [CrossAttnUpBlock2D, (32, 32, 32, 512), {"cross_attention_dim": 32}, (torch.randn(1, 32, 16, 16), (torch.randn(1, 32, 16, 16),), torch.randn(1, 512),), {}, {}, "cuda"],
    [UNetMidBlock2D, (32, 512), {"attention_head_dim": 32}, (torch.randn(1, 32, 16, 16), torch.randn(1, 512)), {}, {}, "cuda"],
    [UpDecoderBlock2D, (32, 32), {"temb_channels": 512}, (torch.randn(1, 32, 16, 16), torch.randn(1, 512)), {}, {}, "cuda"],
    [Decoder, (), {}, (torch.randn(1, 3, 64, 64), ), {}, {}, "cuda"],
    [DownEncoderBlock2D, (32, 32), {}, (torch.randn(1, 32, 16, 16), ), {}, {}, "cuda"],
    [Encoder, (), {}, (torch.randn(1, 3, 64, 64), ), {}, {}, "cuda"],
))  
def test_diffusers_submodules_memory_allocated(model,
                                model_args,
                                model_kwargs,
                                inputs,
                                input_kwargs,
                                dynamic_dims,
                                device):

    if len(model_args) == 0:
        torch_model = model(**model_kwargs)
    else:
        torch_model = model(*model_args, **model_kwargs)
    
    print(f"Running model {torch_model.__class__.__name__}")
    torch_model.eval()
    print("Model eval finished!")
    # breakpoint()
    tmp_folder = f"./temp_files"
    vllmgraph = DiffusionGraphCompiler(torch_model.__class__.__name__, tmp_folder, debug = True)
    vllmgraph.compile(torch_model, inputs, input_kwargs, dynamic_dims = dynamic_dims)
    print("Model compiled!")
    IRdict = vllmgraph.get_graph_dict()
    new_input = []

    #TODO: Need to deal with this anamoly, where tuple of tensors are flattened by the compiler.
    for tensor in inputs:
        if(isinstance(tensor, tuple) or isinstance(tensor, list)):
            new_input.extend([t.to(device) for t in tensor])
        else:
            new_input.append(tensor.to(device))

    reconstructed_model = reconstruct_model(IRdict, f"{tmp_folder}/{torch_model.__class__.__name__}")
    print("Model reconstructed!")

    reconstructed_model["main"].to(device)
    torch_model.to(device)

    new_input = [tensor.to(device) for tensor in new_input]
    inputs = [tensor.to(device) if isinstance(tensor, torch.Tensor) else [t.to(device) for t in tensor] for tensor in inputs]
    
    torch_model(*inputs, **input_kwargs)
    reconstructed_model["main"](*new_input)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]



    with profile(activities=activities, profile_memory=True) as prof2:
        torch.cuda.reset_peak_memory_stats()
        torch_model(*inputs, **input_kwargs)
        peak_original = torch.cuda.max_memory_allocated()

    with profile(activities=activities, profile_memory=True) as prof1:
        torch.cuda.reset_peak_memory_stats()
        reconstructed_model["main"](*new_input)
        peak_diffusion_graph = torch.cuda.max_memory_allocated()

    
    prof1.export_chrome_trace(f"{tmp_folder}/{torch_model.__class__.__name__}/diffusion_graph_trace.json")
    prof2.export_chrome_trace(f"{tmp_folder}/{torch_model.__class__.__name__}/torch_trace.json")
    
    print(f"Diffusion graph allocated {peak_diffusion_graph / (1024 * 1024)} MB vs original model allocated {peak_original / (1024 * 1024)} MB")
    assert peak_diffusion_graph <= peak_original, f"Diffusion graph allocated more memory than original model: {peak_diffusion_graph / (1024 * 1024)} MB vs {peak_original / (1024 * 1024)} MB"
