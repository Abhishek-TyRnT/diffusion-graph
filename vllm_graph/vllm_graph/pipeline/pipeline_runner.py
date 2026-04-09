import torch
import os
import json
from vllm_graph.reconstruct import reconstruct_model
from vllm_graph.model_wrappers import VaeEncoderWrapper, VaeDecoderWrapper, CLIPWrapper, UNetWrapper
from vllm_graph.steppers.stepper import PNDMStepper
from transformers import AutoTokenizer

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
        return AutoTokenizer.from_pretrained(self.tokenizer_name)

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

        self.vae_decoder.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.to(self.device)
        self.stepper.to(self.device)

        print("Pipeline constructed successfully")

    def generate_sample(self):
        return torch.randn(*self.config["latent_shape"], device=self.device)

    def run(self, sample, text_embeddings):

        print("Starting denoising process")
        #TODO: Convert this function into async generator
        for timestep in self.stepper.timesteps:
            print(f"Denoising at timestep {timestep}")
            timestep = torch.tensor([timestep], device=self.device)
            model_output = self.unet(sample, timestep, text_embeddings)
            sample = self.stepper.step(model_output, timestep, sample)
        
        print("Denoising process completed")
        return sample
    
    def generate(self, prompt: str):
        input_tokens = self.tokenizer(prompt, 
                        return_attention_mask=True, 
                        padding="max_length", 
                        truncation=True, 
                        max_length=self.max_length, 
                        return_tensors="pt")
        input_tokens = {k: v.to(self.device) for k, v in input_tokens.items()}
        text_embeddings = self.text_encoder(**input_tokens)
        sample = self.generate_sample()
        sample = self.run(sample, text_embeddings)
        image = self.vae_decoder(sample / self.vae_scaling_factor)
        return image
        
