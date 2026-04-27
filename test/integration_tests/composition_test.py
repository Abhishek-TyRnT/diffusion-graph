import pytest
import torch
import gc
from vllm_graph.reconstruct import reconstruct_model
from vllm_graph.pipeline.pipeline_compiler import DiffusionGraphCompiler
from test_utils import validate_outputs
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.models import UNet2DConditionModel

@pytest.mark.parametrize("text_encoder, text_encoder_class, unet_class, \
 unet_args, unet_kwargs, positive_prompt, negative_prompt, shape, timestep, guidance_scale",(
    ["openai/clip-vit-large-patch14", CLIPTextModel, UNet2DConditionModel, 
        (64, ), {"cross_attention_dim": 768}, "a scenary of a mountain", "", (1, 4, 64, 64), 781, 7.5],
))
def test_composition(text_encoder, text_encoder_class, 
                    unet_class, unet_args, unet_kwargs, 
                    positive_prompt, negative_prompt, shape, timestep, guidance_scale):
    tokenizer = AutoTokenizer.from_pretrained(text_encoder)
    text_encoder_model = text_encoder_class.from_pretrained(text_encoder, attn_implementation = None)
    max_length = text_encoder_model.config.max_position_embeddings
    inputs = tokenizer(positive_prompt, padding="max_length", truncation=True, 
                        max_length=max_length, return_tensors="pt", 
                        return_attention_mask = True)
    negative_inputs = tokenizer(negative_prompt, padding="max_length", truncation=True, 
                        max_length=max_length, return_tensors="pt", 
                        return_attention_mask = True)
    #model(**inputs)
    tmp_folder = f"./temp_files"

    device = "cuda"
    model_name = text_encoder_model.__class__.__name__
    dg_compiler = DiffusionGraphCompiler(model_name,temp_directory = f"{tmp_folder}", debug = True)
    input_kwargs = {"return_dict" : False, "attention_mask" : inputs['attention_mask']}
    dg_compiler.compile(text_encoder_model, (inputs['input_ids'], ), input_kwargs)
    
    compiled_model_dict = dg_compiler.get_graph_dict()
    
    if("compute_pooling_layer" in compiled_model_dict):
        compiled_model_dict.pop("compute_pooling_layer")

    reconstructed_model = reconstruct_model(compiled_model_dict, f"{tmp_folder}/{model_name}")
    #Offloading to target device
    inputs = {key : inputs[key].to(device) for key in inputs}
    negative_inputs = {key : negative_inputs[key].to(device) for key in negative_inputs}
    input_kwargs["attention_mask"] = input_kwargs["attention_mask"].to(device)

    reconstructed_model["main"].to(device)
    
    text_encoder_model = text_encoder_model.to(device)
    #Profiling
    inputs.update(input_kwargs)
    
    normal_text_embedding = text_encoder_model(inputs['input_ids'], input_kwargs['attention_mask'])[0]
    diffusion_graph_text_embedding = reconstructed_model["main"](inputs['input_ids'], input_kwargs['attention_mask'])

    normal_empty_text_embedding = text_encoder_model(**negative_inputs, return_dict = False)[0]
    diffusion_graph_empty_text_embedding = reconstructed_model["main"](negative_inputs['input_ids'], input_kwargs['attention_mask'])

    del text_encoder_model
    del reconstructed_model
    del dg_compiler
    torch.cuda.empty_cache()
    gc.collect()

    unet_model = unet_class(*unet_args, **unet_kwargs)

    model_name = unet_model.__class__.__name__

    sample = torch.randn(*shape)
    sample = torch.cat([sample, sample], dim = 0)

    timestep = torch.tensor([timestep]*2)

    dummy_embeddings = torch.randn(2, *normal_empty_text_embedding.shape[1:])
    inputs = (sample, timestep, dummy_embeddings)
    input_kwargs = {"return_dict" : False}
    dg_compiler = DiffusionGraphCompiler(model_name, temp_directory = f"{tmp_folder}", debug = True)
    dg_compiler.compile(unet_model, inputs, input_kwargs)
    print("Model compiled!")
    IRdict = dg_compiler.get_graph_dict()
    # vllmgraph.store_graph_dict()
    
    dg_embeddings = torch.cat([diffusion_graph_text_embedding, diffusion_graph_empty_text_embedding], dim = 0)
    normal_embeddings = torch.cat([normal_text_embedding, normal_empty_text_embedding], dim = 0)

    new_dg_inputs = (sample.to(device), timestep.to(device), dg_embeddings.to(device))
    new_normal_inputs = (sample.to(device), timestep.to(device), normal_embeddings.to(device))

    del dg_compiler
    gc.collect()
    reconstructed_model = reconstruct_model(IRdict, f"{tmp_folder}/{model_name}")
    print("Model reconstructed!")

    reconstructed_model = {key: model.to(device) for key, model in reconstructed_model.items()}

    with torch.no_grad():
        dg_output = reconstructed_model["main"](*new_dg_inputs)

        del reconstructed_model["main"]
        torch.cuda.empty_cache()
        gc.collect()
        
        unet_model = unet_model.to(device)
        normal_output = unet_model(*new_normal_inputs, return_dict = False)[0]
        del unet_model
        torch.cuda.empty_cache()
        gc.collect()
    
    dg_cond, dg_uncond = torch.chunk(dg_output, 2, dim = 0)
    normal_cond, normal_uncond = torch.chunk(normal_output, 2, dim = 0)

    dg_output = dg_uncond + guidance_scale * (dg_cond - dg_uncond)
    normal_output = normal_uncond + guidance_scale * (normal_cond - normal_uncond)

    validate_outputs(dg_output, normal_output, atol = 1e-6)
    del dg_output, normal_output
    gc.collect()

