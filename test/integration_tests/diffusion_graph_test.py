import pytest
import os
import sys
import json
import torch
from diffusion_graph.reconstruct import reconstruct_model
from diffusion_graph.pipeline.model_compiler import DiffusionGraphCompiler
from diffusion_graph.validator.model_validator import build_validated_engine
from transformers.modeling_outputs import BaseModelOutput
from typing import Dict, List, Optional, Tuple, Union
from torch.export import Dim
from test_utils import validate_outputs


def getmodel_path():
    model_path = os.path.dirname(__file__)
    model_path = os.path.dirname(model_path)
    return model_path

sys.path.append(getmodel_path())

from test_models import *

    

@pytest.mark.parametrize("model, model_args, inputs, dynamic_dims",(
    [Add, (), (torch.randn(224, 10, 3), torch.randn(224, 10, 3)), {}],
     [RSub, (2.,), (torch.randn(224, 10, 3),), {}],
     [LinearModule, (10, 5), (torch.randn(1, 224, 10),), {}],
     [ReluModule, (), (torch.randn(1, 10, 5),), {}],
     [Softmax, (1,), (torch.randn(1, 10, 5),), {}],
     [Transpose, (1, 0), (torch.randn(25, 10),), {}],
    [BatchMatmul, (), (torch.randn(3, 10, 3), torch.randn(3, 3, 10)), {}],
    [AttentionHead, (256, 512, 256), (torch.randn(3, 256, 256), torch.randn(3, 256, 256), torch.randn(3, 256, 256)), {}],
    [LayerNorm, (5, True, True), (torch.randn(3, 256, 5), ), {}],
    [Tanh, (), (torch.randn(3, 256, 1024),), {}],
    [NewGELUActivation, (),  (torch.randn(3, 256, 1024),), {}],
    [Embedding, (2, 3), (torch.tensor([0, 1]), ), {}],
    [Permute, ((0, 2, 1),), (torch.randn(8, 100, 50),), {}],
    [SliceTensorDim1axis, (1, 10, 1), (torch.randn(1, 20),), {}],
    [UnSqueezeOp, (1,), (torch.randn(2, 8),), {}],
    [SqueezeOp, (1,), (torch.randn(2, 8),), {}],
    [Where, (), (torch.randn(5, 1, 8) < 0.5 , torch.rand(5, 1, 8), torch.tensor(5.)), {}],
    [SDPAttention, (0.0,), (torch.randn(3, 256, 256), torch.randn(3, 256, 256), torch.randn(3, 256, 256)), {}],
    [PoolingLayer, (8, ), (torch.randn(3, 8, 8), ), { "inputs" : { 1 : Dim("hidden_size", min = 1, max = 100)}}],
    [Conv2D, (3, 128, 3, 1, 1), (torch.randn(1, 3, 256, 256),), {}],
    [GroupNorm, (4, 16, 1e-5, True), (torch.randn(2, 16, 32, 32),), {}],
    [SiLU, (), (torch.randn(3, 256, 1024),), {}],
    [GeGeLU, (16, 32), (torch.randn(1, 32, 16),), {}],
    [UpsampleNearest2d, (2,), (torch.randn(1, 32, 16, 16),), {}],
    [Sigmoid, (), (torch.randn(1, 32, 16, 16),), {}],
    [SDPA, (), (torch.randn(1, 256, 256), torch.randn(1, 256, 256), torch.randn(1, 256, 256)), {}],
    [SDPA, (), (torch.randn(1, 8, 64, 64), torch.randn(1, 8, 64, 64), torch.randn(1, 8, 64, 64)), {}],

     ))
def test_graph_compiler_python_to_dict(model,
                                       model_args,
                                       inputs,
                                       dynamic_dims):
    
    if len(model_args) == 0:
        torch_model = model()
    else:
        torch_model = model(*model_args)
    
    tmp_folder = f"/tmp"

    model_compiler = DiffusionGraphCompiler(model.__name__, tmp_folder)
    model_compiler.compile(torch_model, inputs, dynamic_dims = dynamic_dims)
    IRdict = model_compiler.get_graph_dict()
    print(json.dumps(IRdict, indent = 2))

@pytest.mark.parametrize("model, model_args, inputs",(
     [Add, (), (torch.randn(224, 10, 3), torch.randn(224, 10, 3))],
     [RSub, (2.,), (torch.randn(224, 10, 3),)],
     [LinearModule, (10, 5), (torch.randn(1, 224, 10),)],
     [ReluModule, (), (torch.randn(1, 10, 5),)],
     [Softmax, (1,), (torch.randn(1, 10, 5),)],
     [Transpose, (1, 0), (torch.randn(25, 10),)],
     [BatchMatmul, (), (torch.randn(3, 10, 3), torch.randn(3, 3, 10))],
     [AttentionHead, (256, 512, 256), (torch.randn(3, 256, 256), torch.randn(3, 256, 256), torch.randn(3, 256, 256))],
     [LayerNorm, (5, True, True), (torch.randn(3, 256, 5),)],
     [Tanh, (), (torch.randn(3, 256, 1024),)],
     [NewGELUActivation, (),  (torch.randn(3, 256, 1024),)],
     [Embedding, (2, 3), (torch.tensor([0, 1]), )],
     [Permute, ((0, 2, 1),), (torch.randn(8, 100, 50),)],
     [SliceTensorDim1axis, (1, 10, 1), (torch.randn(1, 20),)],
     [UnSqueezeOp, (1,), (torch.randn(2, 8),)],
     [SqueezeOp, (1,), (torch.randn(2, 8),)],
     [Where, (), (torch.randn(5, 1, 8) < 0.5 , torch.rand(5, 1, 8), torch.tensor(5.))],
     [SDPAttention, (0.0,), (torch.randn(3, 256, 256), torch.randn(3, 256, 256), torch.randn(3, 256, 256))],
     [Conv2D, (3, 128, 3, 1, 1), (torch.randn(1, 3, 256, 256),)],
     [GroupNorm, (4, 16, 1e-5, True), (torch.randn(2, 16, 32, 32),)],
     [SiLU, (), (torch.randn(3, 256, 1024),)],
     [GeGeLU, (16, 32), (torch.randn(1, 32, 16),)],
     [UpsampleNearest2d, (2,), (torch.randn(1, 32, 16, 16),)],
     [Sigmoid, (), (torch.randn(1, 32, 16, 16),)],
     [SDPA, (), (torch.randn(1, 256, 256), torch.randn(1, 256, 256), torch.randn(1, 256, 256))],
     [SDPA, (), (torch.randn(1, 8, 64, 64), torch.randn(1, 8, 64, 64), torch.randn(1, 8, 64, 64))],
     [PermuteLayerNorm, ((0, 2, 3, 1), (2, 16, 8)), (torch.randn(2, 8, 4, 4),)],
     [PermuteConv2D, ((0, 3, 1, 2), 3, 16, 3, 1, 1), (torch.randn(2, 4, 4, 3),)],

     ))
def test_graph_compiler_to_model(model,
                                       model_args,
                                       inputs):
    
    
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
    diffusion_graph_output = reconstructed_model["main"](*inputs)
    normal_output = torch_model(*inputs)

    assert validate_outputs(diffusion_graph_output, normal_output), f"Test failed validation check"

    
@pytest.mark.parametrize("model, model_args, inputs, dynamic_dims",(
    pytest.param(PoolingLayer, (8, ), (torch.randn(3, 8, 8), ), { "inputs" : { 1 : Dim("hidden_size", min = 1, max = 100)}}, marks=pytest.mark.xfail),
))
def test_graph_compiler_function_partioning_to_model(model,
                                                     model_args,
                                                     inputs,
                                                     dynamic_dims):
    if len(model_args) == 0:
        torch_model = model()
    else:
        torch_model = model(*model_args)
    
    tmp_folder = f"/tmp"

    vllmgraph = DiffusionGraphCompiler(model.__name__, tmp_folder)
    vllmgraph.compile(torch_model, inputs, dynamic_dims = dynamic_dims)
    IRdict = vllmgraph.get_graph_dict()
    reconstructed_model = reconstruct_model(IRdict, f"{tmp_folder}/{torch_model.__class__.__name__}")

    hidden_states = reconstructed_model["main"](*inputs)
    pooling_output = reconstructed_model["compute_pooling_layer"](hidden_states)

    diffusion_graph_output = hidden_states , pooling_output
    normal_output = torch_model(*inputs)

    assert validate_outputs(diffusion_graph_output, normal_output), f"Test failed validation check"


@pytest.mark.parametrize("model, model_args, inputs",(
    [Add, (), (torch.randn(224, 10, 3), torch.randn(224, 10, 3))],
))
def test_model_ops_shapes_and_dtype_validation(model, model_args, inputs):
    if len(model_args) == 0:
        torch_model = model()
    else:
        torch_model = model(*model_args)
    
    tmp_folder = f"./temp_files"

    model_compiler = DiffusionGraphCompiler(torch_model.__class__.__name__, tmp_folder)
    model_compiler.compile(torch_model, inputs)
    IRdict = model_compiler.get_graph_dict()
    model_compiler.store_graph_dict()
    reconstructed_model = reconstruct_model(IRdict, f"{tmp_folder}/{torch_model.__class__.__name__}")
    x = (torch.randn(224, 10, 4), torch.randn(224, 10, 4))
    violations = build_validated_engine(IRdict, reconstructed_model['main'], user_dummy_inputs=inputs)

    assert len(violations) == 0, f"Test failed validation check"