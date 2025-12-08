from transformers.modeling_outputs import BaseModelOutput
import torch


def validate_outputs(vllm_graph_output, regular_output) -> bool:
    if(isinstance(regular_output, tuple)):
        assert len(vllm_graph_output) == len(vllm_graph_output), "No of outputs donot match"
        for (vllm_tensor, tensor) in zip(vllm_graph_output, regular_output):
            if not torch.allclose(vllm_tensor, tensor, atol = 1e-3):
                return False
        
        return True
    if(isinstance(regular_output, BaseModelOutput)):
        return torch.allclose(vllm_graph_output, regular_output.last_hidden_state, atol = 1e-3)
    
    else:
        return torch.allclose(vllm_graph_output, regular_output, atol = 1e-3)