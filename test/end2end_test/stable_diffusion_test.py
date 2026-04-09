import pytest
import torch
from diffusers import StableDiffusionPipeline
from vllm_graph.pipeline.pipeline_compiler import DiffusionPipelineCompiler
from vllm_graph.pipeline.pipeline_runner import DiffusionGraphRunner
import os
import time

def generate_pipe(model_id,):
    
    # Check if CUDA is available, otherwise use CPU    
    print(f"Loading model {model_id}...")
    print("This may take a few minutes on first run as it downloads the model...")
    
    # Load the pipeline
    # Using float16 for GPU to save memory, float32 for CPU
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None
    )
    
    return pipe
    
    # Generate the image
@pytest.mark.parametrize("model_id, model_name, image_shape", (
    ("runwayml/stable-diffusion-v1-5", "stable_diffusion_v1_5", (512, 512)),
))
def test_stable_diffusion(model_id, model_name, image_shape):

    tmp_folder = "./test/examples"
    pipeline = generate_pipe(model_id)
    compiler = DiffusionPipelineCompiler(model_name, tmp_folder, debug=False)
    compiler.compile(pipeline, image_shape)


@pytest.mark.parametrize("model_path, model_name, device, num_inference_steps, tokenizer, prompt", (
    ("test/examples/stable_diffusion_v1_5", "stable_diffusion_v1_5", "cuda", 50, "openai/clip-vit-large-patch14", "a photo of an astronaut riding a horse"),
))
def test_stable_diffusion_inference(model_path, model_name, device, num_inference_steps, tokenizer, prompt):
    
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    artifact_path = os.path.join(root_path, model_path)
    
    runner = DiffusionGraphRunner(artifact_path, device, num_inference_steps, tokenizer)

    runner.load_pipeline()

    start_time = time.perf_counter()
    image = runner.generate(prompt)
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time}")
    
    