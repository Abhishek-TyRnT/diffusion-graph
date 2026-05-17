# Diffusion Graph

> **An MLIR-based diffusion inference engine that efficiently runs diffusion models at scale**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLIR](https://img.shields.io/badge/built%20with-MLIR-orange)](https://mlir.llvm.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17%2B-brightgreen)](https://en.cppreference.com/w/cpp/17)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

---

## Overview

**diffusion-graph** is a high-performance inference engine for diffusion models, built on the [MLIR](https://mlir.llvm.org/) compiler infrastructure. It compiles diffusion model graphs — including UNets, DiT architectures, and VAEs — into optimized, hardware agnostic bytecode which can be run on any hardware.

Unlike general-purpose runtimes, diffusion-graph is architected around the unique computational patterns of diffusion: iterative denoising loops, timestep-conditioned operators, and attention-heavy transformer blocks. This focus enables aggressive fusion, scheduling, and memory planning that generic inference engines miss.

---

## Features

- **MLIR-native compilation** — Torch Models are lowered and compiled to custom designed MLIR dialect called
    diffusion-graph dialect. 
- **Denoising loop awareness** — The compiler understands the iterative structure of diffusion sampling and hoists invariant subgraphs out of the loop.
- **Quantization support** — INT8, FP8, and mixed-precision quantization integrated at the IR level.
- **Scheduler-agnostic** — Works with DDPM, DDIM, DPM-Solver, Euler, and custom samplers out of the box.
- **Python & C++ APIs** — High-level Python bindings for research workflows and deployments

---

## Architecture

```
Model Source (DiffusersPipeline Object)
        │
        ▼
  [Torch-MLIR Frontend]
   - Graph capture & tracing
   - Canonicalization
        │
        ▼
  [diffusion-graph Dialect (MLIR)]
   - Memory optimised passes
   - Operator fusion passes
        │
        ▼
    Compiled Bytecode
        │
        ▼
    Runtime Execution on CUDA
   (Python bindings / C++ API)
```

---

## Getting Started


### Build from Source

```bash
git clone https://github.com/your-org/diffusion-graph.git
cd diffusion-graph
```

Build developer docker image

```
./run_script.sh build
./run_script.sh run
```

Inside Docker container

```

build llvm
build compiler

```


### Install via pip (pre-built wheels)

```bash
pip install diffusion-graph
```

---

## Quick Start
python
```
from diffusers import StableDiffusionPipeline
from diffusion_graph.pipeline.pipeline_compiler import DiffusionPipelineCompiler
from diffusion_graph.pipeline.pipeline_runner import DiffusionGraphRunner

artifact_path = <artifact path>
device = "cuda"
num_inference_steps = 50
tokenizer = "openai/clip-vit-large-patch14"
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None
    )

dg_compiler = DiffusionPipelineCompiler(model_name, tmp_folder, debug=False)
dg_compiler.compile(pipeline, image_shape)

runner = DiffusionGraphRunner(artifact_path, device, num_inference_steps, tokenizer)
runner.load_pipeline()

image = runner.generate(prompt)

```

---

## Supported Models

| Model | Architecture | Status |
|---|---|---|
| Stable diffusion v1-5| UNet | ✅ Stable |

---
---

## Project Structure

```
diffusion-graph/
├── include/            # Public C++ headers
├── lib/
│   ├── Dialect/        # Diffusion MLIR dialect definitions
│   ├── Transforms/     # Optimization passes
│   ├── Conversion/     # Lowering passes to LLVM/CUDA/etc.
│   └── Runtime/        # Engine runtime & memory management
├── python/             # Python bindings (pybind11)
├── tools/              # dg-compile, dg-inspect, dg-bench CLIs
├── test/               # Lit & unit tests
├── benchmarks/         # Reproducible benchmark scripts
└── examples/           # End-to-end usage examples
```

---

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines on submitting passes, dialect extensions, and new backend targets.

---

## Roadmap

- [ ] INT4 weight-only quantization
- [ ] Speculative decoding for consistency models
- [ ] Multi-GPU tensor parallelism
- [ ] WebGPU backend
- [ ] Dynamic resolution & batch size without recompilation
- [ ] ONNX export of compiled engines

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.

---