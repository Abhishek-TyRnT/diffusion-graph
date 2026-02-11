import pytest
import json
import torch
from vllm_graph.reconstruct import vLLMGraph, vLLMGraphModel, reconstruct_model
from transformers import AutoTokenizer, AlbertModel, AlbertForMaskedLM
from test_utils import validate_outputs
# from transformers.models.albert.modeling_albert import AlbertEmbeddings, AlbertSdpaAttention, AlbertLayer, AlbertTransformer
from transformers import AlbertConfig
from torch.profiler import profile, record_function, ProfilerActivity
from torch_mlir.compiler_utils import (
    TensorPlaceholder,
)
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from torch.export import Dim



@pytest.mark.parametrize("model, model_args, inputs, dynamic_dims",(
    [TimestepEmbedding, (16, 32), (torch.randn(1, 32, 16), ), {}],
    [Timesteps, (16, True, 0.1), (torch.randint(0, 1000, (16,)), ), {}],
))
def test_diffusers_submodule_layers(model,
                                model_args,
                                inputs,
                                dynamic_dims):

    if len(model_args) == 0:
        torch_model = model()
    else:
        torch_model = model(*model_args)
    
    torch_model.eval()

    tmp_folder = f"/tmp"
    vllmgraph = vLLMGraph(torch_model.__class__.__name__, tmp_folder)
    vllmgraph.compile(torch_model, inputs)
    IRdict = vllmgraph.get_graph_dict()
    reconstructed_model = reconstruct_model(IRdict)
    vllm_graph_output = reconstructed_model["main"](*inputs)
    normal_output = torch_model(*inputs)

    assert validate_outputs(vllm_graph_output, normal_output), f"Test failed validation check"

@pytest.mark.parametrize("model_name, dummy_input, text, model_class, device",
    (
    ["albert/albert-base-v2",(torch.zeros(1, 100, dtype = torch.int32), torch.zeros(1, 100, dtype = torch.int32)),  "Hello, my dog is cute", AlbertModel, "cuda"],
    ["albert/albert-base-v2",(torch.zeros(1, 100, dtype = torch.int32), torch.zeros(1, 100, dtype = torch.int32)), "Hello, my dog is cute", AlbertForMaskedLM, "cuda"],
     ))
def test_hf_models(model_name, dummy_input, text, model_class, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, attn_implementation = None)
    
    dummy_input_ids = dummy_input[0]
    dummy_position_ids = dummy_input[1]
    inputs = tokenizer(text, return_tensors="pt")
    #model(**inputs)
    tmp_folder = f"./temp_files"

    seq_dim = Dim("seq_len", min = 1, max = model.config.max_position_embeddings - 1)
    dynamic_dims = {
        "input_ids": {1 : seq_dim},
        "position_ids": {1 : seq_dim},
        "return_dict": None
    }
    vllmgraph = vLLMGraph(model_name,temp_directory = tmp_folder, debug = True)
    dummy_input_kwargs = {"return_dict" : False, 'position_ids': dummy_position_ids, }
    vllmgraph.compile(model, (dummy_input_ids, ), dummy_input_kwargs, dynamic_dims = dynamic_dims)
    
    compiled_model_dict = vllmgraph.get_graph_dict()
    
    reconstructed_model = reconstruct_model(compiled_model_dict)
    reconstructed_model = vLLMGraphModel(reconstructed_model, weights_directory = compiled_model_dict['weights_directory'])

    #Offloading to target deivce
    inputs = {key : inputs[key].to(device) for key in inputs}
   
    reconstructed_model.to(device)
    
    model = model.to(device)
    input_kwargs = {"position_ids" : torch.zeros_like(inputs['input_ids']).to(device), "return_dict" : False}
    reconstructed_model(inputs['input_ids'], input_kwargs['position_ids'])
    model(inputs["input_ids"], position_ids = input_kwargs['position_ids'])
    #Profiling
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    inputs.update(input_kwargs)
    
    with profile(activities=activities) as prof2:
        normal_output = model(inputs["input_ids"], **input_kwargs)
    prof2.export_chrome_trace("torch_trace.json")
    
    with profile(activities=activities) as prof1:
        #TODO: Make function splitting optional
        if(hasattr(reconstructed_model, "compute_pooling_layer")):
            hidden_states = reconstructed_model(inputs['input_ids'], input_kwargs['position_ids'])
            pooling_output = reconstructed_model.compute_pooling_layer(hidden_states)
            vllm_graph_output = (hidden_states, pooling_output)
        else:
            vllm_graph_output = reconstructed_model(inputs['input_ids'], input_kwargs['position_ids'])

    prof1.export_chrome_trace("vllm_graph_trace.json")

    assert validate_outputs(vllm_graph_output, normal_output), f"Test failed validation check"