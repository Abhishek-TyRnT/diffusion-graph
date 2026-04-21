import pytest
import torch
import os
import sys
from vllm_graph.pipeline.model_compiler import DiffusionGraphCompiler
from vllm_graph.reconstruct import reconstruct_model

from torch.profiler import profile, record_function, ProfilerActivity


def getmodel_path():
    model_path = os.path.dirname(__file__)
    model_path = os.path.dirname(model_path)
    return model_path

sys.path.append(getmodel_path())

from test_models import *


@pytest.mark.parametrize("model, model_args, inputs, device",(
     [Add, (), (torch.randn(224, 10, 3), torch.randn(224, 10, 3)), "cuda"],
     [RSub, (2.,), (torch.randn(224, 10, 3),), "cuda"],
     [LinearModule, (10, 5), (torch.randn(1, 224, 10),), "cuda"],
     [ReluModule, (), (torch.randn(1, 10, 5),), "cuda"],
     [Softmax, (1,), (torch.randn(1, 10, 5),), "cuda"],
     [Transpose, (1, 0), (torch.randn(25, 10),), "cuda"],
     [BatchMatmul, (), (torch.randn(3, 10, 3), torch.randn(3, 3, 10)), "cuda"],
     [AttentionHead, (256, 512, 256), (torch.randn(3, 256, 256), torch.randn(3, 256, 256), torch.randn(3, 256, 256)), "cuda"],
     [LayerNorm, (5, True, True), (torch.randn(3, 256, 5),), "cuda"],
     [Tanh, (), (torch.randn(3, 256, 1024),), "cuda"],
     [NewGELUActivation, (),  (torch.randn(3, 256, 1024),), "cuda"],
     [Embedding, (2, 3), (torch.tensor([0, 1]), ), "cuda"],
     [Permute, ((0, 2, 1),), (torch.randn(8, 100, 50),), "cuda"],
     pytest.param(SliceTensorDim1axis, (1, 10, 2), (torch.randn(1, 20),), "cuda", marks=pytest.mark.xfail),
     [UnSqueezeOp, (1,), (torch.randn(2, 8),), "cuda"],
     [SqueezeOp, (1,), (torch.randn(2, 8),), "cuda"],
     [Where, (), (torch.randn(5, 1, 8) < 0.5 , torch.rand(5, 1, 8), torch.tensor(5.)), "cuda"],
     [SDPAttention, (0.0,), (torch.randn(3, 256, 256), torch.randn(3, 256, 256), torch.randn(3, 256, 256)), "cuda"],
     [Conv2D, (3, 128, 3, 1, 1), (torch.randn(1, 3, 256, 256),), "cuda"],
     [GroupNorm, (4, 16, 1e-5, True), (torch.randn(2, 16, 32, 32),), "cuda"],
     [SiLU, (), (torch.randn(3, 256, 1024),), "cuda"],
     [GeGeLU, (16, 32), (torch.randn(1, 32, 16),), "cuda"],
     [UpsampleNearest2d, (2,), (torch.randn(1, 32, 16, 16),), "cuda"],
     [Sigmoid, (), (torch.randn(1, 32, 16, 16),), "cuda"],
     [SDPA, (), (torch.randn(1, 256, 256), torch.randn(1, 256, 256), torch.randn(1, 256, 256)), "cuda"],
     [SDPA, (), (torch.randn(1, 8, 64, 64), torch.randn(1, 8, 64, 64), torch.randn(1, 8, 64, 64)), "cuda"],

     ))
def test_graph_compiler_to_model(model,
                                       model_args,
                                       inputs, 
                                       device):
    
    
    if len(model_args) == 0:
        torch_model = model()
    else:
        torch_model = model(*model_args)
    
    tmp_folder = f"./temp_files"

    model_compiler = DiffusionGraphCompiler(torch_model.__class__.__name__, tmp_folder)
    model_compiler.compile(torch_model, inputs)
    model_compiler.store_graph_dict()
    IRdict = model_compiler.get_graph_dict()
    reconstructed_model = reconstruct_model(IRdict, f"{tmp_folder}/{torch_model.__class__.__name__}")
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]


    inputs = [tensor.to(device) for tensor in inputs]
    torch_model.to(device)
    reconstructed_model["main"].to(device)
    with profile(activities=activities, profile_memory=True) as prof2:
        torch.cuda.reset_peak_memory_stats()
        torch_model(*inputs)
        peak_original = torch.cuda.max_memory_allocated()

    with profile(activities=activities, profile_memory=True) as prof1:
        torch.cuda.reset_peak_memory_stats()
        reconstructed_model["main"](*inputs)
        peak_diffusion_graph = torch.cuda.max_memory_allocated()

    
    prof1.export_chrome_trace(f"{tmp_folder}/{torch_model.__class__.__name__}/diffusion_graph_trace.json")
    prof2.export_chrome_trace(f"{tmp_folder}/{torch_model.__class__.__name__}/torch_trace.json")

    assert peak_diffusion_graph <= peak_original, f"Diffusion graph used more memory than original model: {peak_diffusion_graph / (1024 * 1024)} MB vs {peak_original / (1024 * 1024)} MB"

