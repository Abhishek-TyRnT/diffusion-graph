import torch
import os
import json
from diffusion_graph.reconstruct import reconstruct_model
from diffusion_graph.validator.model_validator import build_validated_engine
from diffusion_graph.modelmaps import TYPE_MAP
from diffusion_graph.model_wrappers import VaeEncoderWrapper, VaeDecoderWrapper, CLIPWrapper, UNetWrapper
from diffusion_graph.steppers.stepper import PNDMStepper
from transformers import CLIPTokenizer
from torch import fft
import numpy as np
import math
from tqdm import tqdm
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
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

def project(
        v0: torch.Tensor, # [B, C, H, W]
        v1: torch.Tensor, # [B, C, H, W]
    ):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

def adaptive_projected_guidance(
    pred_cond: torch.Tensor, # [B, C, H, W]
    pred_uncond: torch.Tensor, # [B, C, H, W]
    guidance_scale: float,
    momentum_buffer = None,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
):
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        raise NotImplementedError("Momentum buffer not implemented yet")
    if norm_threshold > 0:
        raise NotImplementedError("Norm threshold not implemented yet")
    
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    return pred_guided

class DiffusionGraphRunner:
    def __init__(self, artifact_directory: str, device:str, tokenizer:str):
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

        self.tokenizer_name = tokenizer
        self.max_length = self.config["input_token_shape"][1]
        self.vae_scaling_factor = self.config["vae_downscaling_factor"]

        self.image_shape = tuple(self.config["image_shape"])
        self.is_pipeline_ready = False

    def load_tokenizer(self):
        return CLIPTokenizer.from_pretrained(self.tokenizer_name)
    
    def _generate_dummy_inputs(self, model_dict):

        tensors = []
        for arg_index in model_dict['main']['entrypoint']:
            arg_info = model_dict['main'][arg_index]
            dtype_str = arg_info["dtype"]
            dtype = TYPE_MAP[dtype_str]

            if dtype == torch.bool:
                tensors.append(torch.randint(0, 2, arg_info["shape"], dtype=torch.bool))
            elif dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                tensors.append(torch.randint(0, 100, arg_info["shape"], dtype=dtype))
            else:
                tensors.append(torch.randn(arg_info["shape"], dtype=dtype))
        return tensors


    def load_model(self, config_path: str, validate_pipeline: bool, **kwargs):

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
        if validate_pipeline:
            new_input = self._generate_dummy_inputs(model_config)
            print("Validating model", model_name)
            build_validated_engine(model_config, model['main'], user_dummy_inputs=new_input)
            print("Validated model", model_name)
        model = WRAPPER_MAP[model_name](model, **kwargs)
        return model
    
    def load_pipeline(self, capture_graph = True, validate_pipeline = True):
        print("Constructing pipeline")

        self.vae_decoder = self.load_model(os.path.join(self.artifact_directory, "vae_decoder/model.json"),
                                            validate_pipeline=validate_pipeline)
        self.vae_encoder = self.load_model(os.path.join(self.artifact_directory, "vae_encoder/model.json"), 
                                            validate_pipeline=validate_pipeline,
                                            vae_scaling_factor=self.vae_scaling_factor)
        self.text_encoder = self.load_model(os.path.join(self.artifact_directory, "text_encoder/model.json"),
                                            validate_pipeline=validate_pipeline)
        self.unet = self.load_model(os.path.join(self.artifact_directory, "unet/model.json"),
                                            validate_pipeline=validate_pipeline)

        self.tokenizer = self.load_tokenizer()

        stepper_config = self.config["stepper_config"]

        diffusers_scheduler_name = stepper_config["_class_name"]
        diffusers_scheduler_class = getattr(diffusers_schedulers, diffusers_scheduler_name)
        diffusers_scheduler = diffusers_scheduler_class.from_config(stepper_config)
        # diffusers_scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        # diffusers_scheduler.to(self.device)

        self.stepper = diffusers_scheduler

        self.vae_decoder.to(self.device)
        self.vae_encoder.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.to(self.device)            

        if capture_graph and self.device == "cuda":
            self.unet.generate_dummy_inputs(self.config['latent_shape'], self.config['hidden_state_shape'], do_classifier_free_guidance=True)
            self.unet.capture_graph()
            
        print("Pipeline constructed successfully")
        self.is_pipeline_ready = True

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
    
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.stepper.timesteps[t_start * self.stepper.order :]
        if hasattr(self.stepper, "set_begin_index"):
            self.stepper.set_begin_index(t_start * self.stepper.order)

        return timesteps, t_start
    
    @torch.inference_mode()
    def decode_latent(self, latent):
        latent = (1 / self.vae_scaling_factor) * latent
        image = self.vae_decoder(latent)
        image = self._post_process_image(image)
        return image

    @torch.inference_mode()
    def encode_image(self, image, strength, num_inference_steps):
        image = self._preprocess_image(image)
        latent = self.vae_encoder(image)
        noise = torch.randn_like(latent)
        timesteps, start_index = self.get_timesteps(num_inference_steps = num_inference_steps, strength = strength, device = self.device)
        latent_timestep = timesteps[:1]
        latent = self.stepper.add_noise(latent, noise, latent_timestep)
        return latent, start_index
    
    def _post_process_image(self, image):
        image = (image *0.5 + 0.5)
        image = image.clip(0, 1)
        image = image[0]
        image = image.permute(1 ,2 , 0)
        image = image.cpu().numpy()
        image = (image * 255).round().astype(np.uint8)
        return image
    
    def _preprocess_image(self, image):
        image = torch.tensor(image, device=self.device)
        image = image.float()
        image = image / 255.0
        image = (image * 2.0) - 1.0
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
        return image
    
    
    def run_denoising_loop(self, sample, 
                        text_embeddings, 
                        uncond_text_embeddings, 
                        guidance_scale, 
                        do_adaptive_guidance, 
                        eta,
                        stream = False,
                        start_index = 0
                        ):
        
        print("Starting denoising process")
        
        
        # Pre-concatenate the text embeddings for efficiency
        multi_batch_text_embeddings = torch.cat([uncond_text_embeddings, text_embeddings], dim=0)

        for timestep in tqdm(self.stepper.timesteps[start_index:], desc="Denoising", unit="steps"):
            sample = self.run_timestep(sample, timestep, multi_batch_text_embeddings, guidance_scale, do_adaptive_guidance, eta)
            if stream:
                yield sample, timestep
        
        if not stream:
            yield sample


    @torch.inference_mode()
    def run_timestep(self, sample, timestep, multi_batch_text_embeddings, guidance_scale, do_adaptive_guidance, eta):

        #TODO: Convert this function into async generator
        multi_batch_sample = torch.cat([sample, sample], dim=0)
        # if hasattr(self.stepper, "scale_model_input"):
        #         multi_batch_sample = self.stepper.scale_model_input(multi_batch_sample, timestep)

        timestep_tensor = torch.tensor([timestep], device=self.device)
        batched_timestep = timestep_tensor.expand(2)
        model_output = self.unet(multi_batch_sample, batched_timestep, multi_batch_text_embeddings)

        uncond_model_output, model_output = model_output.chunk(2, dim=0)

        if do_adaptive_guidance:
            guided_model_output = adaptive_projected_guidance(model_output, uncond_model_output, 
                                                            guidance_scale, eta = 0.2,
                                                            momentum_buffer=None #Momentum currently not supported
                                                            )
        else:
            guided_model_output = (1 - guidance_scale) * uncond_model_output + guidance_scale * model_output
        sample = self.stepper.step(guided_model_output, timestep, sample).prev_sample
        
        return sample
    
    @torch.inference_mode()
    def encode_prompt(self, prompt: str, negative_prompt: str | None = None):
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

        return text_embeddings, uncond_text_embeddings

    
    def generate(self, prompt: str, 
                    image: np.ndarray | None = None, 
                    negative_prompt: str | None = None, 
                    num_inference_steps: int  = 50,
                    guidance_scale: float = 7.5,
                    do_classifier_free_guidance: bool = True,
                    do_adaptive_guidance: bool = False,
                    eta: float = 0.1,
                    strength: float = 0.8 ):


        if do_adaptive_guidance and do_classifier_free_guidance:
            raise ValueError("Adaptive guidance and classifier free guidance cannot be used together")
        
        self.stepper.set_timesteps(num_inference_steps, device=self.device)

        guidance_scale = torch.tensor(guidance_scale, device=self.device)

        text_embeddings, uncond_text_embeddings = self.encode_prompt(prompt, negative_prompt)

        start_index = 0

        if image is None:
            sample = self.generate_sample()
        else:
            assert tuple(image.shape[:-1]) == self.image_shape[2:], f"Image shape {image.shape[:-1]} does not match expected shape {self.image_shape}"
            sample, start_index = self.encode_image(image, strength, num_inference_steps)

        sample, = self.run_denoising_loop(sample, text_embeddings, 
                                    uncond_text_embeddings, guidance_scale, 
                                    do_adaptive_guidance, eta,
                                    start_index = start_index)
        
        image = self.decode_latent(sample)
        return image
    
    def generate_stream(self,
                    file_path: str,
                    prompt: str, 
                    negative_prompt: str | None = None, 
                    num_inference_steps: int  = 50,
                    guidance_scale: float = 7.5,
                    do_classifier_free_guidance: bool = True,
                    do_adaptive_guidance: bool = False,
                    eta: float = 0.1 ):
        
        if do_adaptive_guidance and do_classifier_free_guidance:
            raise ValueError("Adaptive guidance and classifier free guidance cannot be used together")
        
        guidance_scale = torch.tensor(guidance_scale, device=self.device)
        self.stepper.set_timesteps(num_inference_steps, device=self.device)

        text_embeddings, uncond_text_embeddings = self.encode_prompt(prompt, negative_prompt)

        sample = self.generate_sample()
        def add_text(frame, text):
            # frame: numpy array (H, W, 3)
            img = Image.fromarray(frame)

            draw = ImageDraw.Draw(img)

            # Default font
            font = ImageFont.load_default()

            draw.text(
                (10, 10),      # upper-left corner
                text,
                fill=(255, 0, 0),
                font=font
            )

            return np.array(img)
        
        writer = imageio.get_writer(
            file_path,
            fps=24,
            codec="libx264",
            format="FFMPEG"
        )
        for latent, timestep in self.run_denoising_loop(sample, text_embeddings, uncond_text_embeddings, guidance_scale, do_adaptive_guidance, eta, stream=True):
            image = self.decode_latent(latent)
            image = add_text(image, f"Step {timestep}")
            writer.append_data(image)

        writer.close()
