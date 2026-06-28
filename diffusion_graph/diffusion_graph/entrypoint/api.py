import os
import sys
import torch
import logging
from io import BytesIO
from contextlib import asynccontextmanager
from PIL import Image

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
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="The guidance scale for classifier-free guidance.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Retrieve configuration from environment variables
    model_path = os.getenv("DIFFUSION_MODEL_PATH", "temp_files/stable_diffusion_v1_5")
    
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
    
    device = os.getenv("DIFFUSION_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = os.getenv("DIFFUSION_TOKENIZER", "openai/clip-vit-large-patch14")
    
    logger.info("Initializing DiffusionGraphRunner:")
    logger.info(f"  - Model Path: {model_path}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Tokenizer: {tokenizer}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model path {model_path} does not exist!")
        raise RuntimeError(f"Model path {model_path} does not exist.")
    
    try:
        scheduler = DiffusionGraphScheduler(model_path, device, tokenizer)
        app.state.scheduler = scheduler
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

def run_generation(scheduler: DiffusionGraphScheduler, prompt: str, negative_prompt: str | None, steps: int, guidance_scale: float) -> bytes:
    """Helper to run the generation pipeline and convert output to PNG bytes."""
    # Run pipeline generation (it returns a numpy array of shape (H, W, 3) and type uint8)
    req_id = scheduler.submit_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        do_classifier_free_guidance=True
    )

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
    
    try:
        logger.info(f"Received POST /generate request with prompt: '{request.prompt}', steps: {request.steps}, guidance_scale: {request.guidance_scale}")
        img_bytes = run_generation(
            app.state.scheduler,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            guidance_scale=request.guidance_scale
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
    guidance_scale: float = Query(7.5, ge=1.0, le=20.0, description="The guidance scale for classifier-free guidance.")
):
    """
    Generate an image from a text prompt via GET query parameters.
    Returns the generated image as a PNG file.
    """
    if not hasattr(app.state, "scheduler"):
        raise HTTPException(status_code=503, detail="Scheduler is not loaded.")
    
    try:
        logger.info(f"Received GET /generate request with prompt: '{prompt}', steps: {steps}, guidance_scale: {guidance_scale}")
        img_bytes = run_generation(
            app.state.scheduler,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale
        )
        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        logger.exception("Error generating image in GET request")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "pipeline_loaded": hasattr(app.state, "scheduler")}

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Diffusion Graph API Server")
    parser.add_argument("--model-path", type=str, default="temp_files/stable_diffusion_v1_5", help="Path to the diffusion model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    parser.add_argument("--tokenizer", type=str, default="openai/clip-vit-large-patch14", help="Tokenizer name or path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    return parser.parse_args()

if __name__ == "__main__":
    import uvicorn
    
    args = parse_args()
    
    # Expose command line arguments via environment variables so the lifespan context can read them
    os.environ["DIFFUSION_MODEL_PATH"] = args.model_path
    os.environ["DIFFUSION_DEVICE"] = args.device
    os.environ["DIFFUSION_TOKENIZER"] = args.tokenizer
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)
    
    logger.info(f"Starting server at http://{args.host}:{args.port}")
    try:
        # Run uvicorn server. It handles SIGINT/SIGTERM natively.
        uvicorn.run("diffusion_graph.entrypoint.api:app", host=args.host, port=args.port, log_level="info")
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in entrypoint wrapper. Shutting down gracefully...")
        sys.exit(0)
