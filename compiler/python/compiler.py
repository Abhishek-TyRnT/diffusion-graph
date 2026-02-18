from graph_compiler import vLLMGraph
import torch
from torch.nn import Module
from torch_mlir.fx import export_and_import
from torch._decomp import get_decompositions
import os
from torch._decomp import register_decomposition

# Register a no-op decomposition for upsample_nearest2d.vec
from torch.library import impl

def upsample_nearest_decomposed_v2(input, scale_factor=2):
    """
    Using repeat operations
    """
    B, C, H, W = input.shape
    scale = int(scale_factor)
    
    # Repeat each row 'scale' times
    output = input.repeat_interleave(scale, dim=2)  # [B, C, H*scale, W]
    
    # Repeat each column 'scale' times
    output = output.repeat_interleave(scale, dim=3)  # [B, C, H*scale, W*scale]
    
    return output


BACKEND_END_LEGAL_OPS = ["aten.softmax.int", "aten.native_layer_norm", 
                        "aten._softmax", "aten.dropout", 
                        "aten.addmm", "aten.native_group_norm", 
                        "aten.silu", "aten.upsample_nearest2d"]
DECOMPOSITION_OPS = [torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
                    torch.ops.aten._to_copy,
                    torch.ops.aten._unsafe_index.Tensor,
                    torch.ops.aten.index.Tensor_hacked_twin
                    ]
                    
class GraphCompiler:
    def __init__(self, weight_path: str, debug: bool = False):
        self.compiler = vLLMGraph(weight_path)
        self.debug = debug
        self.weight_path = os.path.dirname(weight_path)
        self.backend_legal_ops = BACKEND_END_LEGAL_OPS
        self.decomposition_table = get_decompositions(DECOMPOSITION_OPS)

    def compile(self, model: Module, inputs: list[torch.Tensor]) -> dict:
        torchIR = export_and_import(model, *inputs, output_type="torch", 
                                    backend_legal_ops = self.backend_legal_ops, 
                                    decomposition_table = self.decomposition_table)

        if self.debug:
            with open(f"{self.weight_path}/model.mlir", 'w') as f:
                f.write(str(torchIR))

        IRDict = self.compiler.compile(str(torchIR))
        return IRDict
