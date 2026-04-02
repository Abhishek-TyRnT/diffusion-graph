# Diffusion Engine Runtime

## Overview

The **runtime** is responsible for executing the inference pipeline of the diffusion model. It orchestrates compiled model components (VAE, CLIP, UNet), manages scheduling, and runs the denoising loop to generate the final image.

Unlike the compiler, which produces optimized execution graphs, the runtime focuses purely on **execution, coordination, and data flow**.

---

## Responsibilities

The runtime module handles:

- Input request handling
- Scheduler-driven timestep management
- Execution of compiled model wrappers
- Denoising loop orchestration
- Latent decoding into final image output

---

## High-Level Architecture

```
                     Input Prompt
                        │
                      Scheduler
                        │
  Compiler    Compiler  | Compiler
      │          │      |    │
   CLIP       UNet      |    VAE
 (Wrapper)   (Wrapper)  | (Wrapper)
      │          │      |    │
      └──────┬───┴──────┴────┘
             │          
      Pipeline Orchestrator
             │
        Output Image
```

---

## Core Components

### 1. Scheduler

**Role:**
- Converts input request into a sequence of timesteps
- Controls noise levels and step progression
- Determines number of denoising iterations

**Inputs:**
- Sampling steps
- Scheduler type/config
- Initial noise seed

**Outputs:**
- Timesteps
- Noise scales (sigmas / alphas)

---

### 2. Model Wrappers

Each model is exposed via a runtime wrapper that executes a precompiled graph.

#### CLIP Wrapper
- Encodes text prompt into embeddings
- Called once per request

#### UNet Wrapper
- Predicts noise residuals during denoising
- Called repeatedly inside loop

#### VAE Wrapper
- Decodes final latent into image space
- Called once after denoising

---

### 3. Pipeline Orchestrator

**Role:**
Central execution engine of the runtime.

**Responsibilities:**
- Coordinates calls between CLIP, UNet, and VAE
- Maintains latent state
- Executes denoising loop
- Applies scheduler updates

---

## Execution Flow

### Step 1: Input Processing
- Receive prompt and generation parameters
- Initialize latent tensor (random noise or conditioned)

---

### Step 2: Text Encoding
- Call CLIP wrapper
- Generate text embeddings

---

### Step 3: Denoising Loop

For each timestep `t` from scheduler:

1. Pass latent + timestep + text embeddings → UNet
2. Predict noise residual
3. Update latent using scheduler rule
4. Repeat

---

### Step 4: Decode

- Pass final latent → VAE decoder
- Convert latent → image

---

### Step 5: Output

- Return generated image

---

## Interfaces

### Runtime Entry Point

```python
def generate(
    prompt: str,
    num_steps: int,
    seed: int,
    guidance_scale: float,
    scheduler_config: dict
) -> "Image":
    pass
```

---

### Wrapper Interface

```python
class ModelWrapper:
    def forward(self, *inputs):
        pass
```

---

## Design Principles

### 1. Strict Separation from Compiler
- Runtime never transforms graphs
- Only executes precompiled artifacts

### 2. Stateless Model Execution
- Wrappers are pure execution units
- State is maintained in orchestrator

### 3. Pluggable Scheduler
- Scheduler can be swapped without modifying pipeline logic

### 4. Minimal Overhead Loop
- Denoising loop is optimized for:
  - low latency
  - minimal memory movement

---

## Extensibility

The runtime supports:

- New schedulers (DDIM, Euler, etc.)
- Multiple UNet variants
- Batched inference
- Multi-device execution (future)

---

## Future Improvements

- Async execution for overlapping compute
- KV-cache style reuse for UNet
- Graph-level fusion across steps
- Dynamic step pruning

---

## Summary

The runtime is a **thin but critical orchestration layer** that:

- Drives the diffusion process
- Executes compiled model graphs
- Produces final images efficiently

It is intentionally kept **simple, modular, and execution-focused**, enabling independent evolution of the compiler stack.
