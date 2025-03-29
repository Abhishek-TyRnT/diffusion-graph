import torch

TYPE_MAP = {
    "f32": torch.float32,
    "i32" : torch.int32,
    "i64" : torch.int64 
}

OP_MAP = {
    "vllm_graph.vllm.Relu": torch.nn.functional.relu,
    "vllm_graph.vllm.add": torch.add,
    "vllm_graph.vllm.transpose": torch.transpose,
    "vllm_graph.vllm.matmul": torch.matmul,
    "vllm_graph.vllm.softmax": torch.nn.functional.softmax,
    "vllm_graph.vllm.addmm": torch.addmm,
    "vllm_graph.vllm.view": torch.reshape,
    "vllm_graph.vllm.bmm" : torch.bmm,
    "vllm_graph.vllm.div.scalar": torch.div,
    "vllm_graph.vllm.layer_norm": torch.nn.functional.layer_norm,
    "vllm_graph.vllm.tanh": torch.tanh,
    "vllm_graph.vllm.mul": torch.mul,
    "vllm_graph.vllm.pow": torch.pow,
    "vllm_graph.vllm.embedding": torch.nn.functional.embedding,
    "vllm_graph.vllm.broadcast_to" : torch.broadcast_to,
    "vllm_graph.vllm.permute" : torch.permute,
    "vllm_graph.vllm.index_select" : torch.index_select,
    "vllm_graph.vllm.unsqueeze": torch.unsqueeze,
    "vllm_graph.vllm.squeeze": torch.squeeze

}