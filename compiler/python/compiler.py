from graph_compiler import vLLMGraph
import torch
from torch.nn import Module
from torch_mlir.fx import export_and_import
from torch._decomp import get_decompositions
import os


BACKEND_END_LEGAL_OPS = ["aten.softmax.int", "aten.native_layer_norm", "aten._softmax", "aten.dropout", "aten.addmm", "aten.native_group_norm", "aten.silu"]
DECOMPOSITION_OPS = [torch.ops.aten._scaled_dot_product_flash_attention_for_cpu]
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
