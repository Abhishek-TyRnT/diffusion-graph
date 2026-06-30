import os
import sys
import torch
import logging
import time
from io import BytesIO
from contextlib import asynccontextmanager
from PIL import Image
import argparse
import uvicorn

from fastapi import FastAPI, Response, HTTPException, Query
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("diffusion_graph_api")

# Ensure package is importable if path setup is needed
current_dir = os.path.dirname(os.path.abspath(__file__))
# Check if we can import the pipeline runner; if not, add package roots to sys.path
try:
    from diffusion_graph.scheduler.scheduler import DiffusionGraphScheduler
except ImportError:
    # Walk up to locate the package directory
    pkg_dir = os.path.abspath(os.path.join(current_dir, "../.."))
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    # Also add the workspace root if needed
    workspace_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)
    from diffusion_graph.scheduler.scheduler import DiffusionGraphScheduler

# Input Schema for POST requests
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The text prompt to guide image generation.")
    negative_prompt: str | None = Field(default=None, description="The negative prompt to guide image generation.")
    steps: int = Field(default=50, ge=1, le=500, description="The number of denoising steps.")
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="The guidance scale.")
    do_classifier_free_guidance: bool = Field(default=True, description="Whether to use classifier-free guidance.")
    do_adaptive_guidance: bool = Field(default=False, description="Whether to use adaptive guidance.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Retrieve configuration from app.state.config (set by entrypoint before uvicorn starts)
    # Fall back to environment variables for compatibility, then to hardcoded defaults.
    cfg = getattr(app.state, "config", {})
    model_path = cfg.get("model_path") 
    
    # Resolve relative paths relative to workspace root
    if not os.path.isabs(model_path):
        # Resolve workspace root by traversing up until we find setup.py
        root_dir = current_dir
        while root_dir and root_dir != "/":
            if os.path.exists(os.path.join(root_dir, "setup.py")):
                break
            root_dir = os.path.dirname(root_dir)
        if not root_dir or root_dir == "/":
            root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
        model_path = os.path.abspath(os.path.join(root_dir, model_path))
    
    device = cfg.get("device") 
    tokenizer = cfg.get("tokenizer")
    
    logger.info("Initializing DiffusionGraphScheduler:")
    logger.info(f"  - Model Path: {model_path}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Tokenizer: {tokenizer}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model path {model_path} does not exist!")
        raise RuntimeError(f"Model path {model_path} does not exist.")
    
    try:
        scheduler = DiffusionGraphScheduler(model_path, device, tokenizer)
        app.state.scheduler = scheduler

        while not app.state.scheduler.is_ready():
            time.sleep(5)
        logger.info("Scheduler initialized and ready for requests.")
    except Exception as e:
        logger.exception(f"Failed to initialize scheduler: {e}")
        raise RuntimeError(f"Failed to initialize scheduler: {e}")
        
    yield
    
    logger.info("Shutting down: Releasing model resources...")
    if hasattr(app.state, "scheduler"):
        app.state.scheduler.shutdown()
        del app.state.scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Shutdown cleanup complete.")

app = FastAPI(
    title="Diffusion Graph API Server",
    description="FastAPI endpoint for text2img pipeline execution",
    version="0.1.0",
    lifespan=lifespan
)

def run_generation(scheduler: DiffusionGraphScheduler, prompt: str, negative_prompt: str | None, steps: int, guidance_scale: float, do_classifier_free_guidance: bool = True, do_adaptive_guidance: bool = False) -> bytes:
    """Helper to run the generation pipeline and convert output to PNG bytes."""
    # Run pipeline generation (it returns a numpy array of shape (H, W, 3) and type uint8)
    input_args = (prompt, negative_prompt, steps, guidance_scale, {'do_classifier_free_guidance': do_classifier_free_guidance, 'do_adaptive_guidance': do_adaptive_guidance})
    req_id = scheduler.submit_pipeline(input_args)

    image_numpy = scheduler.receive_by_request_id(req_id)
    if image_numpy is None:
        raise HTTPException(status_code=500, detail="Failed to generate image.")

    pil_img = Image.fromarray(image_numpy)
    img_io = BytesIO()
    pil_img.save(img_io, 'PNG')
    img_io.seek(0)
    return img_io.getvalue()

@app.post("/generate", responses={200: {"content": {"image/png": {}}}})
def generate_image_post(request: GenerateRequest):
    """
    Generate an image from a text prompt via POST request.
    Returns the generated image as a PNG file.
    """
    if not hasattr(app.state, "scheduler"):
        raise HTTPException(status_code=503, detail="Scheduler is not loaded.")
    
    if request.do_classifier_free_guidance and request.do_adaptive_guidance:
        raise HTTPException(status_code=400, detail="do_classifier_free_guidance and do_adaptive_guidance cannot be True at the same time.")
    try:
        logger.info(f"Received POST /generate request with prompt: '{request.prompt}', steps: {request.steps}, guidance_scale: {request.guidance_scale}")
        img_bytes = run_generation(
            app.state.scheduler,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            do_classifier_free_guidance=request.do_classifier_free_guidance,
            do_adaptive_guidance=request.do_adaptive_guidance
        )
        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        logger.exception("Error generating image in POST request")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/generate", responses={200: {"content": {"image/png": {}}}})
def generate_image_get(
    prompt: str = Query(..., description="The text prompt to guide image generation."),
    negative_prompt: str | None = Query(None, description="The negative prompt to guide image generation."),
    steps: int = Query(50, ge=1, le=500, description="The number of denoising steps."),
    guidance_scale: float = Query(7.5, ge=1.0, le=20.0, description="The guidance scale for classifier-free guidance."),
    do_classifier_free_guidance: bool = Query(True, description="Whether to use classifier-free guidance."),
    do_adaptive_guidance: bool = Query(False, description="Whether to use adaptive guidance.")
):
    """
    Generate an image from a text prompt via GET query parameters.
    Returns the generated image as a PNG file.
    """
    if not hasattr(app.state, "scheduler"):
        raise HTTPException(status_code=503, detail="Scheduler is not loaded.")

    if do_classifier_free_guidance and do_adaptive_guidance:
        raise HTTPException(status_code=400, detail="do_classifier_free_guidance and do_adaptive_guidance cannot be True at the same time.")
    try:
        logger.info(f"Received GET /generate request with prompt: '{prompt}', steps: {steps}, guidance_scale: {guidance_scale}")
        img_bytes = run_generation(
            app.state.scheduler,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            do_classifier_free_guidance=do_classifier_free_guidance,
            do_adaptive_guidance=do_adaptive_guidance
        )
        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        logger.exception("Error generating image in GET request")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "pipeline_loaded": hasattr(app.state, "scheduler")}

