from graph_compiler import vLLMGraph
import torch
from torch.nn import Module
from torch_mlir.fx import export_and_import
from torch._decomp import get_decompositions
from torch_mlir._mlir_libs._mlir.ir import DenseResourceElementsAttr
import os
from torch._decomp import register_decomposition
from torch_mlir._mlir_libs._mlir.ir import WalkResult
import gc
# Register a no-op decomposition for upsample_nearest2d.vec
from torch.library import impl
import re

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
        self.weight_path = weight_path
        self.backend_legal_ops = BACKEND_END_LEGAL_OPS
        self.decomposition_table = get_decompositions(DECOMPOSITION_OPS)
        self.ir_path = os.path.join(self.weight_path, "model.mlirbc")

    def compile(self, model: Module, inputs: list[torch.Tensor]) -> dict:
        torchIR = export_and_import(model, *inputs, output_type="torch", 
                                    backend_legal_ops = self.backend_legal_ops, 
                                    decomposition_table = self.decomposition_table)

        print("Completed torch-mlir lowering!")
        if self.debug:
            with open(f"{self.weight_path}/model_debug.mlir", 'w') as f:
                f.write(torchIR.operation.get_asm(large_elements_limit=64))

        torchIR.operation.write_bytecode(self.ir_path)
        del torchIR
        gc.collect()

        print("Starting diffusion graph compilation!")
        IRDict = self.compiler.compile(self.ir_path)
        print("Completed graph compilation!")
        if not self.debug:
            os.remove(self.ir_path)
            
        return IRDict
