import pytest
import os
import sys
import json
import torch
from vllm_graph.reconstruct import vLLMGraph, reconstruct_model
from transformers.modeling_outputs import BaseModelOutput
from typing import Dict, List, Optional, Tuple, Union
from torch.export import Dim


def getmodel_path():
    model_path = os.path.dirname(__file__)
    model_path = os.path.dirname(model_path)
    return model_path

sys.path.append(getmodel_path())

from test_models import *

def validate_outputs(vllm_graph_output, regular_output) -> bool:
    if(isinstance(regular_output, tuple)):
        assert len(vllm_graph_output) == len(vllm_graph_output), "No of outputs donot match"
        for (vllm_tensor, tensor) in zip(vllm_graph_output, regular_output):
            if not torch.allclose(vllm_tensor, tensor, atol = 1e-3):
                return False
        
        return True
    if(isinstance(regular_output, BaseModelOutput)):
        return torch.allclose(vllm_graph_output[0], regular_output.last_hidden_state, atol = 1e-3)
    
    else:
        return torch.allclose(vllm_graph_output, regular_output, atol = 1e-3)

    

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
    [SliceTensorDim1axis, (1, 10, 2), (torch.randn(1, 20),), {}],
    [UnSqueezeOp, (1,), (torch.randn(2, 8),), {}],
    [SqueezeOp, (1,), (torch.randn(2, 8),), {}],
    [Where, (), (torch.randn(5, 1, 8) < 0.5 , torch.rand(5, 1, 8), torch.tensor(5.)), {}],
    [SDPAttention, (0.0,), (torch.randn(3, 256, 256), torch.randn(3, 256, 256), torch.randn(3, 256, 256)), {}],
    [PoolingLayer, (8, ), (torch.randn(3, 8, 8), ), { "inputs" : { 1 : Dim("hidden_size", min = 1, max = 100)}}],
    [Conv2D, (3, 128, 3, 1, 1), (torch.randn(1, 3, 256, 256),), {}],
    [GroupNorm, (4, 16, 1e-5, True), (torch.randn(2, 16, 32, 32),), {}],
    [SiLU, (), (torch.randn(3, 256, 1024),), {}],
    [GeGeLU, (16, 32), (torch.randn(1, 32, 16),), {}],

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

    vllmgraph = vLLMGraph(model.__name__, tmp_folder)
    vllmgraph.compile(torch_model, inputs, dynamic_dims = dynamic_dims)
    IRdict = vllmgraph.get_graph_dict()
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
     [SliceTensorDim1axis, (1, 10, 2), (torch.randn(1, 20),)],
     [UnSqueezeOp, (1,), (torch.randn(2, 8),)],
     [SqueezeOp, (1,), (torch.randn(2, 8),)],
     [Where, (), (torch.randn(5, 1, 8) < 0.5 , torch.rand(5, 1, 8), torch.tensor(5.))],
     [SDPAttention, (0.0,), (torch.randn(3, 256, 256), torch.randn(3, 256, 256), torch.randn(3, 256, 256))],
     [Conv2D, (3, 128, 3, 1, 1), (torch.randn(1, 3, 256, 256),)],
     [GroupNorm, (4, 16, 1e-5, True), (torch.randn(2, 16, 32, 32),)],
     [SiLU, (), (torch.randn(3, 256, 1024),)],
     [GeGeLU, (16, 32), (torch.randn(1, 32, 16),)],
     ))
def test_graph_compiler_to_model(model,
                                       model_args,
                                       inputs):
    
    
    if len(model_args) == 0:
        torch_model = model()
    else:
        torch_model = model(*model_args)
    
    tmp_folder = f"/tmp"

    vllmgraph = vLLMGraph(torch_model.__class__.__name__, tmp_folder)
    vllmgraph.compile(torch_model, inputs)
    # vllmgraph.store_graph_dict()
    IRdict = vllmgraph.get_graph_dict()
    reconstructed_model = reconstruct_model(IRdict)
    vllm_graph_output = reconstructed_model["main"](*inputs)
    normal_output = torch_model(*inputs)

    assert validate_outputs(vllm_graph_output, normal_output), f"Test failed validation check"

    
@pytest.mark.parametrize("model, model_args, inputs, dynamic_dims",(
    [PoolingLayer, (8, ), (torch.randn(3, 8, 8), ), { "inputs" : { 1 : Dim("hidden_size", min = 1, max = 100)}}],
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

    vllmgraph = vLLMGraph(model.__name__, tmp_folder)
    vllmgraph.compile(torch_model, inputs, dynamic_dims = dynamic_dims)
    IRdict = vllmgraph.get_graph_dict()
    reconstructed_model = reconstruct_model(IRdict)

    hidden_states = reconstructed_model["main"](*inputs)
    pooling_output = reconstructed_model["compute_pooling_layer"](hidden_states)

    vllm_graph_output = hidden_states , pooling_output
    normal_output = torch_model(*inputs)

    assert validate_outputs(vllm_graph_output, normal_output), f"Test failed validation check"

