import torch
from diffusers import StableDiffusionPipeline
from vllm_graph.model_wrappers import MethodWrapper
from vllm_graph.pipeline.model_compiler import DiffusionGraphCompiler

import gc
import os
import json

class DiffusionPipelineCompiler:
    def __init__(self, pipeline_name, artifact_directory: str, debug: bool = False, **kwargs):
        self.name = pipeline_name

        assert artifact_directory is not None, "Artifact directory is required"
        self.artifact_directory = f"{artifact_directory}/{pipeline_name}"
        
        if not os.path.exists(self.artifact_directory):
            os.makedirs(self.artifact_directory)
        
        self.debug = debug

    def compile(self, pipeline: StableDiffusionPipeline, image_shape: tuple):

        vae = pipeline.vae
        downscale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        latent_shape = (1, vae.config.latent_channels, image_shape[0] // downscale_factor, image_shape[1] // downscale_factor)
        dummy_latent = torch.randn(latent_shape)
        #TODO: Add support for compiler VAE encoder.
        print("Compiling VAE Decoder")
        decoder = MethodWrapper(vae, "decode")
        decoder_compiler = DiffusionGraphCompiler("vae_decoder", self.artifact_directory, self.debug)
        decoder_compiler.compile(decoder, (dummy_latent, ), input_kwargs = {"return_dict": False})
        decoder_compiler.store_graph_dict()

        del decoder_compiler
        gc.collect()

        text_encoder = pipeline.text_encoder

        input_token_shape = (1, text_encoder.config.max_position_embeddings)
        hidden_state_shape = (1, text_encoder.config.max_position_embeddings, text_encoder.config.hidden_size)

        print("Compiling Text Encoder")
        text_encoder_compiler = DiffusionGraphCompiler("text_encoder", self.artifact_directory, self.debug)
        text_encoder_compiler.compile(text_encoder, (torch.randint(0, 1000, input_token_shape), torch.randint(0, 2, input_token_shape)), input_kwargs = {"return_dict": False})
        # print(text_encoder_compiler.get_graph_dict())
        text_encoder_compiler.store_graph_dict()

        del text_encoder_compiler
        gc.collect()

        unet = pipeline.unet
        print("Compiling UNet")
        unet_compiler = DiffusionGraphCompiler("unet", self.artifact_directory, self.debug)
        unet_compiler.compile(unet, (dummy_latent, torch.tensor(1, dtype=torch.long), torch.randn(hidden_state_shape)), input_kwargs = {"return_dict": False})
        unet_compiler.store_graph_dict()

        del unet_compiler
        gc.collect()

        print("Pipeline compilation finished!")

        config = {
            "stepper_config" : dict(pipeline.scheduler.config), 
            "pipeline_name" : self.name,
            "artifact_directory" : self.artifact_directory,
            "image_shape" : image_shape,
            "latent_shape" : latent_shape,
            "input_token_shape" : input_token_shape,
            "hidden_state_shape" : hidden_state_shape,
        }

        with open(os.path.join(self.artifact_directory, "config.json"), "w") as f:
            json.dump(config, f, indent = 2)

        
                
