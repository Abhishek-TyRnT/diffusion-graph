from transformers.modeling_outputs import BaseModelOutput
import torch


def validate_outputs(vllm_graph_output, regular_output, atol = 1e-3) -> bool:
    if(isinstance(regular_output, tuple)):
        if (len(vllm_graph_output) != len(regular_output)):
            new_regular_output = []
            for tensor in regular_output:
                if(isinstance(tensor, tuple)):
                    new_regular_output.extend(tensor)
                else:
                    new_regular_output.append(tensor)
            regular_output = new_regular_output
        
        assert len(vllm_graph_output) == len(regular_output), "No of outputs donot match"
        for (vllm_tensor, tensor) in zip(vllm_graph_output, regular_output):
            if not torch.allclose(vllm_tensor, tensor, atol = atol):
                return False
        
        return True
    if(isinstance(regular_output, BaseModelOutput)):
        return torch.allclose(vllm_graph_output, regular_output.last_hidden_state, atol = atol)
    
    else:
        return torch.allclose(vllm_graph_output, regular_output, atol = atol)