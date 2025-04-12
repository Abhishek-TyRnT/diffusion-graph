import pytest
import os
import sys
import json
import torch
from vllm_graph.reconstruct import vLLMGraph
from transformers.modeling_outputs import BaseModelOutput
from typing import Dict, List, Optional, Tuple, Union

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
        #print(vllm_graph_output[0].shape, regular_output.shape)
        return torch.allclose(vllm_graph_output[0], regular_output, atol = 1e-3)

    

@pytest.mark.parametrize("model, model_args, inputs",(
    [Add, (), (torch.randn(224, 10, 3), torch.randn(224, 10, 3))],
     [RSub, (2.,), (torch.randn(224, 10, 3),)],
     [LinearModule, (10, 5), (torch.randn(1, 224, 10),)],
     [ReluModule, (), (torch.randn(1, 10, 5),)],
     [Softmax, (1,), (torch.randn(1, 10, 5),)],
     [Transpose, (1, 0), (torch.randn(25, 10),)],
    [BatchMatmul, (), (torch.randn(3, 10, 3), torch.randn(3, 3, 10))],
    [AttentionHead, (256, 512, 256), (torch.randn(3, 256, 256), torch.randn(3, 256, 256), torch.randn(3, 256, 256))],
    [LayerNorm, (5, True, True), (torch.randn(3, 256, 5), )],
    [Tanh, (), (torch.randn(3, 256, 1024),)],
    [NewGELUActivation, (),  (torch.randn(3, 256, 1024),)],
    [Embedding, (2, 3), (torch.tensor([0, 1]), )],
    [Permute, ((0, 2, 1),), (torch.randn(8, 100, 50),)],
    [SliceTensorDim1axis, (1, 10, 2), (torch.randn(1, 20),)],
    [UnSqueezeOp, (1,), (torch.randn(2, 8),)],
    [SqueezeOp, (1,), (torch.randn(2, 8),)],
    [Where, (), (torch.randn(5, 1, 8) < 0.5 , torch.rand(5, 1, 8), torch.tensor(5.))],
     ))
def test_graph_compiler_python_to_dict(model,
                                       model_args,
                                       inputs):
    
    if len(model_args) == 0:
        torch_model = model()
    else:
        torch_model = model(*model_args)
    
    tmp_folder = f"/tmp"

    vllmgraph = vLLMGraph(model.__name__, tmp_folder)
    vllmgraph.compile(torch_model, inputs)
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
    
    reconstructed_model = vllmgraph.reconstruct()
    vllm_graph_output = reconstructed_model(*inputs)
    normal_output = torch_model(*inputs)

    assert validate_outputs(vllm_graph_output, normal_output), f"Test failed validation check"
    

