# DEPRECATED

# vLLM-Graph Design

vllm-graph generates a vllm supported torch fx Graph to run using vllm runtime. vllm-graph provides flexibility to run almost any model using the vllm runtime irrespective of the format of the model (ex :- pytorch, tflite, onnx-model).

vllm-graph first takes in model and converts into torch Dialect in torch-mlir. the torch Dialect is then converted to vllmir, a custom Dialect developed specifically to run on vllm. This ir is then converted into torch.fx.Graph to finally run on vllm. 


![vllm-graph-design](images/graph%20design.png)


vllm-graph bypasses the need to run only hugging-face model provided in vllm. 

# vllm design

The vllm has class structure as shown below

![vllm-desgin](images/vllm%20design.png)

The `ModelRunner` class loads the model or downloads if necessary from the hugging-face repo, and then run subsequently on the said hardware.

vllm-graph overrides the `load_model` method in `ModelRunner` to bypass this need and can send desired model in vllm to run using kernels given the vllm repo. 