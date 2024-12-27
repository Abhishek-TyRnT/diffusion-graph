import pytest
import os
import sys
import json
import torch
from vllm_graph.reconstruct import vLLMGraph
def getmodel_path():
    model_path = os.path.dirname(__file__)
    model_path = os.path.dirname(model_path)
    return model_path

sys.path.append(getmodel_path())

from test_models import *

@pytest.mark.parametrize("model, model_args, inputs",
    ([Add, (), (torch.randn(224, 10, 3), torch.randn(224, 10, 3))],
     [LinearModule, (10, 5), (torch.randn(1, 224, 10),)],
     [ReluModule, (), (torch.randn(1, 10, 5))],
     [Softmax, (1,), (torch.randn(1, 10, 5))],
     [Transpose, (1, 0), (torch.randn(25, 10),)],
     [AttentionHead, (256, 512, 256), (torch.randn(3, 256, 256), torch.randn(3, 256, 256), torch.randn(3, 256, 256))]
     ))
def test_graph_compiler_python_to_dict(model,
                                       model_args,
                                       inputs):
    
    if len(model_args) == 0:
        torch_model = model()
    else:
        torch_model = model(*model_args)
    
    tmp_folder = f"/tmp/{torch_model.__class__.__name__}"

    vllmgraph = vLLMGraph(tmp_folder)
    vllmgraph.compile(torch_model, inputs)
    IRdict = vllmgraph.get_graph_dict()
    print(json.dumps(IRdict, indent = 2))

@pytest.mark.parametrize("model, model_args, inputs",
    ([Add, (), (torch.randn(224, 10, 3), torch.randn(224, 10, 3))],
     [LinearModule, (10, 5), (torch.randn(1, 224, 10),)],
     [ReluModule, (), (torch.randn(1, 10, 5),)],
     [Softmax, (1,), (torch.randn(1, 10, 5),)],
     [Transpose, (1, 0), (torch.randn(25, 10),)]
     ))
def test_graph_compiler_to_model(model,
                                       model_args,
                                       inputs):
    
    if len(model_args) == 0:
        torch_model = model()
    else:
        torch_model = model(*model_args)
    
    tmp_folder = f"/tmp/{torch_model.__class__.__name__}"

    vllmgraph = vLLMGraph(tmp_folder)
    vllmgraph.compile(torch_model, inputs)

    reconstructed_model = vllmgraph.reconstruct()

    vllm_graph_output = reconstructed_model(*inputs)
    normal_output = torch_model(*inputs)

    assert torch.allclose(vllm_graph_output, normal_output, atol = 1e-3), f"Test failed validation check"
    

    