import pytest
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from diffusion_graph.pipeline.pipeline_compiler import DiffusionPipelineCompiler
from diffusion_graph.pipeline.pipeline_runner import DiffusionGraphRunner
import os
import time



def plot_timestep_stats(stats_map: dict, save_path: str) -> None:
    """Plot per-timestep statistics and save the figure to *save_path*.

    Args:
        stats_map: A dictionary whose keys are timestep values (numeric) and
            whose values are dictionaries containing the keys:
                - ``"mean"``     – mean of the distribution at that timestep
                - ``"std"``      – standard deviation
                - ``"kurtosis"`` – kurtosis
                - ``"kl"``       – KL divergence
        save_path: Absolute or relative file path (including filename and
            extension, e.g. ``"plots/stats.png"``) where the figure will be
            saved.  Parent directories are created automatically if they do
            not already exist.
    """
    # Sort timesteps so the x-axis is ordered.
    timesteps = sorted(stats_map.keys(), reverse=True)

    means     = [stats_map[t]["mean"]     for t in timesteps]
    stds      = [stats_map[t]["std"]      for t in timesteps]
    kurtoses  = [stats_map[t]["kurtosis"] for t in timesteps]
    kls       = [stats_map[t]["kl"]       for t in timesteps]
    maxs      = [stats_map[t]["max"]      for t in timesteps]
    mins      = [stats_map[t]["min"]      for t in timesteps]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Per-Timestep Statistics", fontsize=16, fontweight="bold")

    plot_configs = [
        (axes[0, 0], means,    "Mean",     "tab:blue"),
        (axes[0, 1], stds,     "Std Dev",  "tab:orange"),
        (axes[1, 0], kurtoses, "Kurtosis", "tab:green"),
        (axes[1, 1], kls,      "KL Divergence", "tab:red"),
        (axes[2, 0], maxs,     "Max",      "tab:purple"),
        (axes[2, 1], mins,     "Min",      "tab:brown"),
    ]

    for ax, values, title, color in plot_configs:
        ax.plot(timesteps, values, marker="o", linewidth=2,
                markersize=4, color=color)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Timestep", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Ensure the parent directory exists.
    parent_dir = os.path.dirname(save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Stats plot saved to: {save_path}")

def plot_timestep_freq_stats(stats_map: dict, save_path: str) -> None:
    """Plot per-timestep statistics and save the figure to *save_path*.

    Args:
        stats_map: A dictionary whose keys are timestep values (numeric) and
            whose values are dictionaries containing the keys:
                - ``"mean"``     – mean of the distribution at that timestep
                - ``"std"``      – standard deviation
                - ``"kurtosis"`` – kurtosis
                - ``"kl"``       – KL divergence
        save_path: Absolute or relative file path (including filename and
            extension, e.g. ``"plots/stats.png"``) where the figure will be
            saved.  Parent directories are created automatically if they do
            not already exist.
    """
    # Sort timesteps so the x-axis is ordered.
    timesteps = sorted(stats_map.keys(), reverse=True)

    energy     = [stats_map[t]["energy"]     for t in timesteps]
    flatness      = [stats_map[t]["flatness"]      for t in timesteps]
    hf_ratio  = [stats_map[t]["hf_ratio"] for t in timesteps]

    fig, axes = plt.subplots(1, 3, figsize=(14, 10))
    fig.suptitle("Per-Timestep Statistics", fontsize=16, fontweight="bold")

    plot_configs = [
        (axes[0], energy,    "Energy",     "tab:blue"),
        (axes[1], flatness,  "Flatness",  "tab:orange"),
        (axes[2], hf_ratio,  "HF Ratio", "tab:green"),
    ]

    for ax, values, title, color in plot_configs:
        ax.plot(timesteps, values, marker="o", linewidth=2,
                markersize=4, color=color)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Timestep", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Ensure the parent directory exists.
    parent_dir = os.path.dirname(save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Stats plot saved to: {save_path}")

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

    tmp_folder = "./temp_files"
    pipeline = generate_pipe(model_id)
    compiler = DiffusionPipelineCompiler(model_name, tmp_folder, debug=False)
    compiler.compile(pipeline, image_shape)


@pytest.mark.parametrize("model_path, model_name, device, num_inference_steps, tokenizer, prompt, negative_prompt, extra_kwargs", (
    ("temp_files/stable_diffusion_v1_5", "stable_diffusion_v1_5", "cuda", 50, 
            "openai/clip-vit-large-patch14", "an astronaut riding a horse", 
            "oil painting, water color, drawing", {"guidance_scale": 7.5}),
    ("temp_files/stable_diffusion_v1_5", "stable_diffusion_v1_5", "cuda", 50, 
        "openai/clip-vit-large-patch14", "a scenary of a mountain", 
        "oil painting, water color, drawing", {"eta": 0.5, "do_adaptive_guidance": True, 
                                            "guidance_scale": 8.5, "do_classifier_free_guidance": False}),
))
def test_stable_diffusion_inference(model_path, model_name, 
                                    device, num_inference_steps, 
                                    tokenizer, prompt, 
                                    negative_prompt, extra_kwargs):
    
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    artifact_path = os.path.join(root_path, model_path)
    
    runner = DiffusionGraphRunner(artifact_path, device, tokenizer)
    runner.load_pipeline()

    start_time = time.perf_counter()
    image = runner.generate(prompt, negative_prompt, num_inference_steps, **extra_kwargs)
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time}")

    plt.imsave(f"{artifact_path}/output.png", image)
    
    