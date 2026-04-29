import torch
import os
import json
from vllm_graph.reconstruct import reconstruct_model
from vllm_graph.model_wrappers import VaeEncoderWrapper, VaeDecoderWrapper, CLIPWrapper, UNetWrapper
from vllm_graph.steppers.stepper import PNDMStepper
from transformers import CLIPTokenizer
from torch import fft
import numpy as np
import math

from diffusers import schedulers as diffusers_schedulers

WRAPPER_MAP = {
    "vae_encoder": VaeEncoderWrapper,
    "vae_decoder": VaeDecoderWrapper,
    "text_encoder": CLIPWrapper,
    "unet": UNetWrapper
}

StepperMap = {
    "PNDMScheduler": PNDMStepper
}


#TODO: Add support for adaptive projected guidance
# class MomentumBuffer:
#     def __init__(self, momentum: float):
#         self.momentum = momentum
#         self.running_average = 0
#     def update(self, update_value: torch.Tensor):
#         new_average = self.momentum * self.running_average
#         self.running_average = update_value + new_average

# def project(
#         v0: torch.Tensor, # [B, C, H, W]
#         v1: torch.Tensor, # [B, C, H, W]
#     ):
#     dtype = v0.dtype
#     v0, v1 = v0.double(), v1.double()
#     v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
#     v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
#     v0_orthogonal = v0 - v0_parallel
#     return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

# def adaptive_projected_guidance(
#     pred_cond: torch.Tensor, # [B, C, H, W]
#     pred_uncond: torch.Tensor, # [B, C, H, W]
#     guidance_scale: float,
#     momentum_buffer: MomentumBuffer = None,
#     eta: float = 1.0,
#     norm_threshold: float = 0.0,
# ):
#     diff = pred_cond - pred_uncond
#     if momentum_buffer is not None:
#         momentum_buffer.update(diff)
#         diff = momentum_buffer.running_average
#     if norm_threshold > 0:
#         ones = torch.ones_like(diff)
#         diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
#         scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
#         diff = diff * scale_factor
#     diff_parallel, diff_orthogonal = project(diff, pred_cond)
#     normalized_update = diff_orthogonal + eta * diff_parallel
#     pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
#     return pred_guided

class DiffusionGraphRunner:
    def __init__(self, artifact_directory: str, device:str, num_inference_steps:int, tokenizer:str):
        self.artifact_directory = artifact_directory
        self.graph_dict = None
        self.weights = None

        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        assert os.path.exists(self.artifact_directory), f"Artifact directory {self.artifact_directory} does not exist"

        self.config_path = os.path.join(artifact_directory, "config.json")

        assert os.path.isfile(self.config_path), f"Config file does not exist at location {self.config_path}"
        with open(self.config_path, "r") as f:
            self.config = json.load(f)
        
        self.device = device
        self.num_inference_steps = num_inference_steps

        self.tokenizer_name = tokenizer
        self.max_length = self.config["input_token_shape"][1]
        self.vae_scaling_factor = self.config["vae_downscaling_factor"]

    def load_tokenizer(self):
        return CLIPTokenizer.from_pretrained(self.tokenizer_name, local_files_only=True #TODO: Remove this when we have a proper way to handle model loading
        )

    def load_model(self, config_path: str,):

        assert os.path.isfile(config_path), f"Config file does not exist at location {config_path}"
        with open(config_path, "r") as f:
            model_config = json.load(f)
        
        #The CLIP Model has pooling layer component which as of now has 
        #no use. It is kept for possible future use. It might be aggresively 
        #pruned from the model in future.
        if(model_config.get("compute_pooling_layer", None) is not None):
            del model_config["compute_pooling_layer"]
        
        model_name = model_config["model_name"]

        config_dir = os.path.dirname(config_path)

        print(f"Loading model {model_name} ...")
        model = reconstruct_model(model_config, config_dir)
        model = WRAPPER_MAP[model_name](model)
        return model
    
    def load_pipeline(self):
        print("Constructing pipeline")

        self.vae_decoder = self.load_model(os.path.join(self.artifact_directory, "vae_decoder/model.json"))
        self.text_encoder = self.load_model(os.path.join(self.artifact_directory, "text_encoder/model.json"))
        self.unet = self.load_model(os.path.join(self.artifact_directory, "unet/model.json"))

        self.tokenizer = self.load_tokenizer()

        stepper_config = self.config["stepper_config"]

        diffusers_scheduler_name = stepper_config["_class_name"]
        diffusers_scheduler_class = getattr(diffusers_schedulers, diffusers_scheduler_name)
        diffusers_scheduler = diffusers_scheduler_class.from_config(stepper_config)
        diffusers_scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        # diffusers_scheduler.to(self.device)

        self.stepper = diffusers_scheduler
        # stepper_class = StepperMap[diffusers_scheduler_name]
        # self.stepper = stepper_class(diffusers_scheduler, self.num_inference_steps)

        self.vae_decoder.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.to(self.device)

        
        print("Pipeline constructed successfully")

    def generate_sample(self):
        x = torch.randn(*self.config["latent_shape"], device=self.device) * self.stepper.init_noise_sigma
        return x

    def stats(self, x):
        x = x.float().view(-1)

        mean = x.mean()
        var = x.var(unbiased=False)
        std = var.sqrt()

        kurt = ((x - mean)**4).mean() / (var**2 + 1e-8)

        kl = 0.5 * (mean**2 + var - torch.log(var + 1e-8) - 1)

        return {
            "mean": mean.item(),
            "std": std.item(),
            "kurtosis": kurt.item(),
            "kl": kl.item(),
            "max": x.max().item(),
            "min": x.min().item(),
        }
    
    def fft_stats(self, eps):
            x = eps.float()

            # ---- FFT ----
            X = fft.fft2(x, norm="ortho")
            power = (X.real**2 + X.imag**2)  # (B, C, H, W)

            # ---- Basic energy ----
            total_energy = power.mean().item()

            # ---- Spectral flatness ----
            power_flat = power.flatten(1) + 1e-12
            geo_mean = torch.exp(torch.mean(torch.log(power_flat), dim=1))
            arith_mean = torch.mean(power_flat, dim=1)
            flatness = (geo_mean / arith_mean).mean().item()

            # ---- High-frequency energy ratio ----
            B, C, H, W = power.shape
            hf_region = power[:, :, H//4:3*H//4, W//4:3*W//4]
            hf_energy = hf_region.mean().item()
            hf_ratio = hf_energy / (total_energy + 1e-12)

            log = {
                "energy": total_energy,
                "flatness": flatness,
                "hf_ratio": hf_ratio,
            }
            return log

    @torch.inference_mode()
    def run(self, sample, text_embeddings, uncond_text_embeddings, guidance_scale):

        print("Starting denoising process")
        multi_batch_text_embeddings = torch.cat([uncond_text_embeddings, text_embeddings], dim=0)
        # multi_batch_text_embeddings = multi_batch_text_embeddings.contiguous()
        #TODO: Convert this function into async generator
        mb = None
        for timestep in self.stepper.timesteps:
            print(f"Denoising at timestep {timestep}")
            multi_batch_sample = torch.cat([sample, sample], dim=0)
            # multi_batch_sample = multi_batch_sample.contiguous()
            # if hasattr(self.stepper, "scale_model_input"):
            #         multi_batch_sample = self.stepper.scale_model_input(multi_batch_sample, timestep)

            timestep_tensor = torch.tensor([timestep], device=self.device)
            batched_timestep = timestep_tensor.expand(2)
            model_output = self.unet(multi_batch_sample, batched_timestep, multi_batch_text_embeddings)

            uncond_model_output, model_output = model_output.chunk(2, dim=0)

            # guided_model_output = adaptive_projected_guidance(model_output, uncond_model_output, 
            #                                                 guidance_scale, eta = 0.1,
            #                                                 momentum_buffer=mb)
            # eps_diff = model_output - uncond_model_output
            guided_model_output = (1 - guidance_scale) * uncond_model_output + guidance_scale * model_output
            sample = self.stepper.step(guided_model_output, timestep.item(), sample).prev_sample
        
        print("Denoising process completed")
        return sample
    
    def generate(self, prompt: str, negative_prompt: str | None = None, guidance_scale: float = 7.5):

        guidance_scale = torch.tensor(guidance_scale, device=self.device)


        input_tokens = self.tokenizer(prompt, 
                        return_attention_mask=True, 
                        padding="max_length", 
                        truncation=True, 
                        max_length=self.max_length, 
                        return_tensors="pt")
        
        input_tokens = {k: v.to(self.device) for k, v in input_tokens.items()}
        text_embeddings = self.text_encoder(**input_tokens)

        if negative_prompt is None:
            negative_prompt = ""

        negative_input_tokens = self.tokenizer(negative_prompt, 
                        return_attention_mask=True, 
                        padding="max_length", 
                        truncation=True, 
                        max_length=self.max_length, 
                        return_tensors="pt")
        negative_input_tokens = {k: v.to(self.device) for k, v in negative_input_tokens.items()}

        uncond_text_embeddings = self.text_encoder(**negative_input_tokens)

        text_embeddings = text_embeddings.to(self.device)
        uncond_text_embeddings = uncond_text_embeddings.to(self.device)
        sample = self.generate_sample()
        sample = self.run(sample, text_embeddings, uncond_text_embeddings, guidance_scale)
        
        sample = (1 / self.vae_scaling_factor) * sample
        image = self.vae_decoder(sample)
        image = (image *0.5 + 0.5)
        image = image.clip(0, 1)
        image = image[0]
        image = image.permute(1 ,2 , 0)
        image = image.cpu().numpy()
        image = (image * 255).round().astype(np.uint8)
        return image
        
