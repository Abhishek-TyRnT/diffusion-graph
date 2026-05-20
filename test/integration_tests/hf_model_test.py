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
from transformers.models.clip.modeling_clip import (CLIPTextEmbeddings,
                                        CLIPAttention, 
                                        CLIPMLP, 
                                        CLIPEncoderLayer, 
                                        CLIPEncoder)
# from transformers.models.albert.modeling_albert import AlbertEmbeddings, AlbertSdpaAttention, AlbertLayer, AlbertTransformer
from transformers import AlbertConfig, CLIPTextConfig
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
    [UpDecoderBlock2D, (32, 32), {"temb_channels": 512}, (torch.randn(1, 32, 16, 16), torch.randn(1, 512)), {}, {}],
    [Decoder, (), {}, (torch.randn(1, 3, 64, 64), ), {}, {}],
    [DownEncoderBlock2D, (32, 32), {}, (torch.randn(1, 32, 16, 16), ), {}, {}],
    [Encoder, (), {}, (torch.randn(1, 3, 64, 64), ), {}, {}],
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
    
    device = "cuda"
    torch.backends.cudnn.deterministic = True

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

    # print(reconstructed_model["main"].print_readable())

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(activities=activities, profile_memory=True) as prof2:
        torch.cuda.reset_peak_memory_stats()
        normal_output = torch_model(*inputs, **input_kwargs)
        peak_original = torch.cuda.max_memory_allocated()

    with profile(activities=activities, profile_memory=True) as prof1:
        torch.cuda.reset_peak_memory_stats()
        diffusion_graph_output = reconstructed_model["main"](*new_input)
        peak_vllm = torch.cuda.max_memory_allocated()

    
    print(f"Peak memory for original model: {peak_original / (1024 * 1024)} MB")
    print(f"Peak memory for vllm graph: {peak_vllm / (1024 * 1024)} MB")
    prof1.export_chrome_trace(f"{tmp_folder}/{torch_model.__class__.__name__}/diffusion_graph_trace.json")
    prof2.export_chrome_trace(f"{tmp_folder}/{torch_model.__class__.__name__}/torch_trace.json")
    print("Outputs validated!")
    assert validate_outputs(diffusion_graph_output, normal_output, atol=1e-6), f"Test failed validation check"

@pytest.mark.parametrize("model, model_args, model_kwargs, inputs, input_kwargs",(
        (CLIPTextEmbeddings, (CLIPTextConfig(),), {}, (torch.randint(0, 1000, (1, 77)),), {}),
        (CLIPAttention, (CLIPTextConfig(),), {}, (torch.randn(1, 32, 512), ), {}),
        (CLIPMLP, (CLIPTextConfig(),), {}, (torch.randn(1, 32, 512), ), {}),
        (CLIPEncoder, (CLIPTextConfig(),), {}, (torch.randn(1, 32, 512),), {}),
    ))
def test_hf_submodules(model, model_args, model_kwargs, inputs, input_kwargs):
    if len(model_args) == 0:
        torch_model = model(**model_kwargs)
    else:
        torch_model = model(*model_args, **model_kwargs)
    
    torch_model.eval()
    print(torch_model)
    print("Model eval finished!")
    # breakpoint()
    tmp_folder = f"./temp_files"
    vllmgraph = DiffusionGraphCompiler(torch_model.__class__.__name__, tmp_folder, debug = True)
    vllmgraph.compile(torch_model, inputs, input_kwargs, dynamic_dims = {}, )
    print("Model compiled!")
    IRdict = vllmgraph.get_graph_dict()
    new_input = []

    #TODO: Need to deal with this anamoly, where tuple of tensors are flattened by the compiler.
    for tensor in inputs:
        if(isinstance(tensor, tuple) or isinstance(tensor, list)):
            new_input.extend(tensor)
        else:
            new_input.append(tensor)

    reconstructed_model = reconstruct_model(IRdict, f"{tmp_folder}/{torch_model.__class__.__name__}")
    print("Model reconstructed!")
    
    diffusion_graph_output = reconstructed_model["main"](*new_input)
    normal_output = torch_model(*inputs, **input_kwargs)
    print("Outputs validated!")
    assert validate_outputs(diffusion_graph_output, normal_output), f"Test failed validation check"

@pytest.mark.parametrize("model, model_args, model_kwargs, inputs, input_kwargs, method",
    (
        [AutoencoderKL, (), {}, (torch.randn(1, 4, 64, 64), ), {'return_dict': False}, "decode"],
        [AutoencoderKL, (), {}, (torch.randn(1, 3, 64, 64), ), {}, "_encode"],
    ))
def test_diffusers_vae(model, 
                model_args, 
                model_kwargs, 
                inputs, 
                input_kwargs, 
                method):
    if len(model_args) == 0:
        torch_model = model(**model_kwargs)
    else:
        torch_model = model(*model_args, **model_kwargs)

    torch_model = MethodWrapper(torch_model, method)
    torch_model.eval()
    print("Model eval finished!")
    # breakpoint()
    tmp_folder = f"/tmp"
    vllmgraph = DiffusionGraphCompiler(torch_model.__class__.__name__, tmp_folder)
    vllmgraph.compile(torch_model, inputs, input_kwargs)
    print("Model compiled!")
    IRdict = vllmgraph.get_graph_dict()
    new_input = []

    #TODO: Need to deal with this anamoly, where tuple of tensors are flattened by the compiler.
    for tensor in inputs:
        if(isinstance(tensor, tuple) or isinstance(tensor, list)):
            new_input.extend(tensor)
        else:
            new_input.append(tensor)

    reconstructed_model = reconstruct_model(IRdict, f"{tmp_folder}/{torch_model.__class__.__name__}")
    print("Model reconstructed!")
    device = "cuda"
    reconstructed_model["main"].to(device)
    torch_model.to(device)

    new_input = [tensor.to(device) for tensor in new_input]
    inputs = [tensor.to(device) if isinstance(tensor, torch.Tensor) else [t.to(device) for t in tensor] for tensor in inputs]

    diffusion_graph_output = reconstructed_model["main"](*new_input)
    normal_output = torch_model(*inputs, **input_kwargs)
    print("Outputs validated!")
    assert validate_outputs(diffusion_graph_output, normal_output, atol=1e-6), f"Test failed validation check"

@pytest.mark.parametrize("model_name, text, model_class, device",
    (
    # pytest.param("albert/albert-base-v2",(torch.zeros(1, 100, dtype = torch.int32), torch.zeros(1, 100, dtype = torch.int32)),  "Hello, my dog is cute", AlbertModel, "cuda", marks = pytest.mark.xfail()),
    # pytest.param("albert/albert-base-v2",(torch.zeros(1, 100, dtype = torch.int32), torch.zeros(1, 100, dtype = torch.int32)), "Hello, my dog is cute", AlbertForMaskedLM, "cuda", marks = pytest.mark.xfail()),
    ["openai/clip-vit-large-patch14", "Hello, my dog is cute", CLIPTextModel, "cuda"],
    ["openai/clip-vit-large-patch14", "", CLIPTextModel, "cuda"],
    ))
def test_hf_models(model_name, text, model_class, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, attn_implementation = None)
    max_length = model.config.max_position_embeddings
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt", return_attention_mask = True)
    #model(**inputs)
    tmp_folder = f"./temp_files"

    vllmgraph = DiffusionGraphCompiler(model_name,temp_directory = tmp_folder, debug = True)
    input_kwargs = {"return_dict" : False, "attention_mask" : inputs['attention_mask']}
    vllmgraph.compile(model, (inputs['input_ids'], ), input_kwargs)
    
    compiled_model_dict = vllmgraph.get_graph_dict()
    
    if("compute_pooling_layer" in compiled_model_dict):
        compiled_model_dict.pop("compute_pooling_layer")

    reconstructed_model = reconstruct_model(compiled_model_dict, f"{tmp_folder}/{model_name}")
    #Offloading to target device
    inputs = {key : inputs[key].to(device) for key in inputs}
    input_kwargs["attention_mask"] = input_kwargs["attention_mask"].to(device)
    reconstructed_model["main"].to(device)
    
    model = model.to(device)
    model(inputs["input_ids"], **input_kwargs)
    #Profiling
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    inputs.update(input_kwargs)
    
    with profile(activities=activities) as prof2:
        normal_output = model(inputs["input_ids"], **input_kwargs)[0]
    prof2.export_chrome_trace("torch_trace.json")
    
    with profile(activities=activities) as prof1:
        #TODO: Make function splitting optional
        if(hasattr(reconstructed_model, "compute_pooling_layer")):
            hidden_states = reconstructed_model["main"](inputs['input_ids'], input_kwargs['attention_mask'])
            # pooling_output = reconstructed_model.compute_pooling_layer(hidden_states)
            diffusion_graph_output = (hidden_states, pooling_output)
        else:
            diffusion_graph_output = reconstructed_model["main"](inputs['input_ids'], input_kwargs['attention_mask'])

    prof1.export_chrome_trace("diffusion_graph_trace.json")

    assert validate_outputs(diffusion_graph_output, normal_output, atol=1e-6), f"Test failed validation check"


@pytest.mark.parametrize("model, model_args, model_kwargs, inputs, input_kwargs, dynamic_dims, device",(
    # [UNet2DModel, (64,), {}, (torch.randn(1, 3, 64, 64), torch.randint(0, 100, (1,))), {'return_dict': False}, {}, "cuda"],
    [UNet2DConditionModel, (64,), {"cross_attention_dim" : 768}, (torch.randn(2, 4, 64, 64), torch.randint(0, 100, (2,)), torch.randn(2, 77, 768)), {'return_dict': False}, {}, "cuda"],
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
    model_name = torch_model.__class__.__name__
    print("Model eval finished!")

    tmp_folder = f"./temp_files"
    vllmgraph = DiffusionGraphCompiler(torch_model.__class__.__name__, tmp_folder, debug = True)
    vllmgraph.compile(torch_model, inputs, input_kwargs, dynamic_dims = dynamic_dims)
    print("Model compiled!")
    IRdict = vllmgraph.get_graph_dict()
    # vllmgraph.store_graph_dict()
    new_input = []
    for tensor in inputs:
        if(isinstance(tensor, tuple) or isinstance(tensor, list)):
            new_input.extend(tensor.to(device))
        else:
            new_input.append(tensor.to(device))

    del vllmgraph
    gc.collect()
    reconstructed_model = reconstruct_model(IRdict, f"{tmp_folder}/{torch_model.__class__.__name__}")
    print("Model reconstructed!")

    reconstructed_model = {key: model.to(device) for key, model in reconstructed_model.items()}

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with torch.no_grad():
        with profile(activities=activities, profile_memory=True) as prof1:

            diffusion_graph_output = reconstructed_model["main"](*new_input)

        del reconstructed_model["main"]
        torch.cuda.empty_cache()
        gc.collect()
        
        torch_model = torch_model.to(device)
        with torch.no_grad():
            with profile(activities=activities, profile_memory=True) as prof2:
                normal_output = torch_model(*new_input, **input_kwargs)
    del torch_model
    gc.collect()
    prof1.export_chrome_trace(f"{tmp_folder}/{model_name}/diffusion_graph_trace.json")
    prof2.export_chrome_trace(f"{tmp_folder}/{model_name}/torch_trace.json")
    # breakpoint()
    print("Outputs validated!")
    assert validate_outputs(diffusion_graph_output, normal_output[0], atol = 1e-5), f"Test failed validation check"
