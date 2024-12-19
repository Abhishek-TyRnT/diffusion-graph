import torch

TYPE_MAP = {
    "f32": torch.float32,
    "i32" : torch.int32, 
}

OP_MAP = {
    "vllm_graph.vllm.Relu": torch.nn.functional.relu,
    "vllm_graph.vllm.add": torch.add,
    "vllm_graph.vllm.transpose": torch.transpose,
    "vllm_graph.vllm.matmul": torch.matmul,
    "vllm_graph.vllm.softmax": torch.nn.functional.softmax,
    "vllm_graph.vllm.addmm": torch.addmm,
    "vllm_graph.vllm.view": torch.reshape
}