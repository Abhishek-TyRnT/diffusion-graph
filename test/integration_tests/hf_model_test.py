import pytest
import json
import torch
from vllm_graph.reconstruct import vLLMGraph, vLLMGraphModel, reconstruct_model
from transformers import AutoTokenizer, AlbertModel, AlbertForMaskedLM
from test_utils import validate_outputs
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward, BasicTransformerBlock
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.unets.unet_2d_blocks import (CrossAttnDownBlock2D, 
                                        CrossAttnUpBlock2D, DownBlock2D, 
                                        UpBlock2D, UNetMidBlock2D)

from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.downsampling import Downsample2D
from diffusers.models.upsampling import Upsample2D
# from transformers.models.albert.modeling_albert import AlbertEmbeddings, AlbertSdpaAttention, AlbertLayer, AlbertTransformer
from transformers import AlbertConfig
from torch.profiler import profile, record_function, ProfilerActivity
from torch_mlir.compiler_utils import (
    TensorPlaceholder,
)
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from torch.export import Dim
import gc



@pytest.mark.parametrize("model, model_args, model_kwargs, inputs, input_kwargs, dynamic_dims",(
    [TimestepEmbedding, (16, 32), {}, (torch.randn(1, 32, 16), ), {}, {}],
    [Timesteps, (16, True, 0.1), {}, (torch.randint(0, 1000, (16,)), ), {}, {}],
    [Attention, (16,), {}, (torch.randn(1, 32, 16), ), {}, {}],
    [FeedForward, (16,), {}, (torch.randn(1, 32, 16), ), {}, {}],
    [BasicTransformerBlock, (16, 8, 16,), {}, (torch.randn(1, 32, 16), ), {}, {}],
    [Transformer2DModel, (16, 88, 32, 32), {}, (torch.randn(1, 32, 16, 16), ), {'return_dict': False}, {}],
    [ResnetBlock2D, (), {'in_channels': 32}, (torch.randn(1, 32, 16, 16), torch.randn(1, 512)), {}, {}],
    [CrossAttnDownBlock2D, (32, 32, 512), {"cross_attention_dim": 128}, (torch.randn(1, 32, 16, 16), torch.randn(1, 512), torch.randn(1, 16, 128)), {}, {}],
    [Downsample2D, (32, ), {'use_conv': True}, (torch.randn(1, 32, 16, 16), ), {}, {}],
    [DownBlock2D, (32, 32, 512), {}, (torch.randn(1, 32, 16, 16), torch.randn(1, 512)), {}, {}],
    [Upsample2D, (32, ), {'use_conv': True}, (torch.randn(1, 32, 16, 16), ), {}, {}],
    [UpBlock2D, (32, 32, 32, 512), {}, (torch.randn(1, 32, 16, 16), (torch.randn(1, 32, 16, 16),), torch.randn(1, 512)), {}, {}],
    [CrossAttnUpBlock2D, (32, 32, 32, 512), {"cross_attention_dim": 32}, (torch.randn(1, 32, 16, 16), (torch.randn(1, 32, 16, 16),), torch.randn(1, 512),), {}, {}],
    [UNetMidBlock2D, (32, 512), {"attention_head_dim": 32}, (torch.randn(1, 32, 16, 16), torch.randn(1, 512)), {}, {}],
))
def test_diffusers_submodule_layers(model,
                                model_args,
                                model_kwargs,
                                inputs,
                                input_kwargs,
                                dynamic_dims):

    if len(model_args) == 0:
        torch_model = model(**model_kwargs)
    else:
        torch_model = model(*model_args, **model_kwargs)
    
    torch_model.eval()
    print("Model eval finished!")
    # breakpoint()
    tmp_folder = f"/tmp"
    vllmgraph = vLLMGraph(torch_model.__class__.__name__, tmp_folder)
    vllmgraph.compile(torch_model, inputs, input_kwargs, dynamic_dims = dynamic_dims)
    print("Model compiled!")
    IRdict = vllmgraph.get_graph_dict()
    new_input = []

    #TODO: Need to deal with this anamoly, where tuple of tensors are flattened by the compiler.
    for tensor in inputs:
        if(isinstance(tensor, tuple) or isinstance(tensor, list)):
            new_input.extend(tensor)
        else:
            new_input.append(tensor)

    reconstructed_model = reconstruct_model(IRdict)
    print("Model reconstructed!")
    vllm_graph_output = reconstructed_model["main"](*new_input)
    normal_output = torch_model(*inputs, **input_kwargs)
    print("Outputs validated!")
    assert validate_outputs(vllm_graph_output, normal_output), f"Test failed validation check"

@pytest.mark.parametrize("model_name, dummy_input, text, model_class, device",
    (
    pytest.param("albert/albert-base-v2",(torch.zeros(1, 100, dtype = torch.int32), torch.zeros(1, 100, dtype = torch.int32)),  "Hello, my dog is cute", AlbertModel, "cuda", marks = pytest.mark.xfail()),
    pytest.param("albert/albert-base-v2",(torch.zeros(1, 100, dtype = torch.int32), torch.zeros(1, 100, dtype = torch.int32)), "Hello, my dog is cute", AlbertForMaskedLM, "cuda", marks = pytest.mark.xfail()),
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


@pytest.mark.parametrize("model, model_args, model_kwargs, inputs, input_kwargs, dynamic_dims, device",(
    [UNet2DModel, (64,), {}, (torch.randn(1, 3, 64, 64), torch.randint(0, 100, (1,))), {'return_dict': False}, {}, "cuda"],
    [UNet2DConditionModel, (64,), {}, (torch.randn(1, 4, 64, 64), torch.randint(0, 100, (1,)), torch.randn(1, 16, 1280)), {'return_dict': False}, {}, "cuda"],
))
def test_full_diffusers_model(model, 
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
    
    torch_model.eval()
    print("Model eval finished!")

    tmp_folder = f"./temp_files"
    vllmgraph = vLLMGraph(torch_model.__class__.__name__, tmp_folder, debug = True)
    vllmgraph.compile(torch_model, inputs, input_kwargs, dynamic_dims = dynamic_dims)
    print("Model compiled!")
    IRdict = vllmgraph.get_graph_dict()
    vllmgraph.store_graph_dict()
    new_input = []
    for tensor in inputs:
        if(isinstance(tensor, tuple) or isinstance(tensor, list)):
            new_input.extend(tensor.to(device))
        else:
            new_input.append(tensor.to(device))

    del vllmgraph
    gc.collect()
    reconstructed_model = reconstruct_model(IRdict)
    print("Model reconstructed!")

    reconstructed_model = {key: model.to(device) for key, model in reconstructed_model.items()}

    vllm_graph_output = reconstructed_model["main"](*new_input)
    del reconstructed_model
    gc.collect()
    torch_model = torch_model.to(device)
    normal_output = torch_model(*new_input, **input_kwargs)
    print("Outputs validated!")
    assert validate_outputs(vllm_graph_output, normal_output, atol = 1e-2), f"Test failed validation check"
