import torch
import os
import json
from vllm_graph.reconstruct import reconstruct_model
from vllm_graph.model_wrappers import VaeEncoderWrapper, VaeDecoderWrapper, CLIPWrapper, UNetWrapper
from vllm_graph.steppers.stepper import PNDMStepper
from transformers import CLIPTokenizer

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

class DiffusionGraphRunner:
    def __init__(self, artifact_directory: str, device:str, num_inference_steps:int, tokenizer:str):
        self.artifact_directory = artifact_directory
        self.graph_dict = None
        self.weights = None

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

        stepper_class = StepperMap[diffusers_scheduler_name]
        self.stepper = stepper_class(diffusers_scheduler, self.num_inference_steps)

        self.vae_decoder.to("cpu")
        self.text_encoder.to("cpu")
        self.unet.to(self.device)
        self.stepper.to(self.device)

        self.empty_input_tokens = self.tokenizer("", 
                        return_attention_mask=True, 
                        padding="max_length", 
                        truncation=True, 
                        max_length=self.max_length, 
                        return_tensors="pt")
        # self.empty_input_tokens = {k: v.to(self.device) for k, v in self.empty_input_tokens.items()}

        print("Pipeline constructed successfully")

    def generate_sample(self):
        return torch.randn(*self.config["latent_shape"], device=self.device)

    def run(self, sample, text_embeddings, uncond_text_embeddings, guidance_scale):

        print("Starting denoising process")
        multi_batch_text_embeddings = torch.cat([text_embeddings, uncond_text_embeddings], dim=0)
        #TODO: Convert this function into async generator
        for timestep in self.stepper.timesteps:
            print(f"Denoising at timestep {timestep}")
            multi_batch_sample = torch.cat([sample, sample], dim=0)
            batched_timestep = torch.tensor([timestep] * 2, device=self.device)
            timestep = torch.tensor([timestep], device=self.device)
            model_output = self.unet(multi_batch_sample, batched_timestep, multi_batch_text_embeddings)

            model_output, uncond_model_output = model_output.chunk(2, dim=0)
            model_output = uncond_model_output + guidance_scale * (model_output - uncond_model_output)

            sample = self.stepper.step(model_output, timestep, sample)
        
        print("Denoising process completed")
        return sample
    
    def generate(self, prompt: str, guidance_scale: float = 7.5):
        input_tokens = self.tokenizer(prompt, 
                        return_attention_mask=True, 
                        padding="max_length", 
                        truncation=True, 
                        max_length=self.max_length, 
                        return_tensors="pt")
        # input_tokens = {k: v.to(self.device) for k, v in input_tokens.items()}
        text_embeddings = self.text_encoder(**input_tokens)
        uncond_text_embeddings = self.text_encoder(**self.empty_input_tokens)

        text_embeddings = text_embeddings.to(self.device)
        uncond_text_embeddings = uncond_text_embeddings.to(self.device)
        sample = self.generate_sample()
        sample = self.run(sample, text_embeddings, uncond_text_embeddings, guidance_scale)
        sample = sample.to("cpu")
        image = self.vae_decoder(sample / self.vae_scaling_factor)
        image = image + 1
        image = image.clip(0, 1)
        image = image[0]
        image = image.permute(1 ,2 , 0)
        return image
        
