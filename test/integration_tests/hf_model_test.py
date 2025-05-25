import pytest
import json
import torch
from vllm_graph.reconstruct import vLLMGraph, vLLMGraphModel
from transformers import AutoTokenizer, AlbertModel, AlbertForMaskedLM
from test_utils import validate_outputs
from transformers.models.albert.modeling_albert import AlbertEmbeddings, AlbertSdpaAttention, AlbertLayer, AlbertTransformer
from transformers import AlbertConfig
from torch.profiler import profile, record_function, ProfilerActivity


@pytest.mark.parametrize("Model, inputs, input_kwargs", (
    [AlbertEmbeddings, (torch.randint(0, 100, (8, 512)),), {}],
    [AlbertSdpaAttention, (torch.randn(8, 512, 4096),), {}],
    [AlbertLayer, (torch.randn(1, 8, 4096),), {}],
    [AlbertTransformer, (torch.randn(1, 8, 128),), {"return_dict": False}],
))
def test_hf_model_layer(Model,
                        inputs,
                        input_kwargs):
    config = AlbertConfig()
    model = Model(config)
    model.eval()
    tmp_folder = f"/tmp"

    vllmgraph = vLLMGraph(model.__class__.__name__, tmp_folder, )
    vllmgraph.compile(model, inputs, input_kwargs)
    reconstructed_model = vllmgraph.reconstruct()
    normal_output = model(*inputs)
    vllm_graph_output = reconstructed_model(*inputs)

    assert validate_outputs(vllm_graph_output, normal_output), f"Test failed validation check"


@pytest.mark.parametrize("model_name, text, model_class, device",
    (
     ["albert/albert-base-v2", "Hello, my dog is cute", AlbertModel, "cuda"],
    ["albert/albert-base-v2", "Hello, my dog is cute", AlbertForMaskedLM, "cuda"],
     ))
def test_hf_models(model_name, text, model_class, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, attn_implementation = None)
    inputs = tokenizer(text, return_tensors="pt")
    
    #model(**inputs)
    tmp_folder = f"/tmp"

    vllmgraph = vLLMGraph(model_name,temp_directory = tmp_folder, debug = True)
    position_ids = torch.zeros_like(inputs['input_ids'], dtype = torch.int32)
    input_kwargs = {"return_dict" : False, 'position_ids': position_ids}
    vllmgraph.compile(model, (inputs["input_ids"], ), input_kwargs)
    reconstructed_model = vllmgraph.reconstruct()
    reconstructed_model = vLLMGraphModel(reconstructed_model)
    input_kwargs["position_ids"] = input_kwargs["position_ids"].to(device)

    #Offloading to target deivce
    inputs = {key : inputs[key].to(device) for key in inputs}
    reconstructed_model.to(device)
    # reconstructed_model.device = device
    model = model.to(device)
    reconstructed_model(inputs['input_ids'], input_kwargs['position_ids'])
    model(inputs["input_ids"], position_ids = input_kwargs['position_ids'])
    #Profiling
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    inputs.update(input_kwargs)
    
    with profile(activities=activities) as prof2:
        normal_output = model(inputs["input_ids"], **input_kwargs)
    prof2.export_chrome_trace("torch_trace.json")
    
    with profile(activities=activities) as prof1:
        vllm_graph_output = reconstructed_model(inputs['input_ids'], input_kwargs['position_ids'], )
    prof1.export_chrome_trace("vllm_graph_trace.json")


    assert validate_outputs(vllm_graph_output, normal_output), f"Test failed validation check"

