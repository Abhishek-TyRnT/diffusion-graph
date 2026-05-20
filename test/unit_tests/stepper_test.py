import pytest
import torch
from diffusion_graph.steppers.stepper import PNDMStepper
from diffusers import PNDMScheduler

@pytest.mark.parametrize("diffusers_scheduler, diffusion_graph_scheduler, kwargs, shape, num_of_steps",(
    (PNDMScheduler, PNDMStepper, {"skip_prk_steps": True}, (1, 4, 64, 64), 100),
))
def test_steppers_validation(diffusers_scheduler, diffusion_graph_scheduler, kwargs, shape, num_of_steps):
    diffusers_scheduler = diffusers_scheduler(num_train_timesteps=1000, **kwargs)
    diffusion_graph_scheduler = diffusion_graph_scheduler(diffusers_scheduler, num_of_steps)
    
    def get_model_output(shape, timestep, seed):
        generator = torch.Generator().manual_seed(seed)
        model_output = torch.randn(shape, generator=generator)
        
        # Add slight timestep-dependent bias (earlier timesteps have larger magnitude)
        # This mimics real diffusion model behavior
        scale = 1.0 + (timestep / 1000.0) * 0.5
        model_output = model_output * scale
        return model_output

    def generate_sample(
        shape, 
        seed = None
    ):
        """Generate initial noisy sample."""
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
            
        return torch.randn(shape, generator=generator)

    diffusers_sample = generate_sample(shape, seed=42)
    diffusion_graph_sample = diffusers_sample.clone()
        
    for i, t in enumerate(diffusers_scheduler.timesteps):
        # Generate model output based on current sample state
        model_output = get_model_output(
            shape, 
            t.item(), 
            seed=1000 + i
            )
            
        diffusers_sample = diffusers_scheduler.step(
            model_output,
            t,
            diffusers_sample,
            return_dict=False
        )[0]
            
        diffusion_graph_sample = diffusion_graph_scheduler.step(
                model_output.clone(),
                t,
                diffusion_graph_sample,
            )
        
        assert torch.allclose(diffusers_sample, diffusion_graph_sample, atol=1e-3), f"Samples do not match at timestep {t}"
    