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
        self.weight_path = os.path.dirname(weight_path)
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

        #TODO: Converting it to string and parsing the string in C++ is slow 
        # and memory intensive. Need to bypass parsing for end2end flow
        print("Converting TorchIRModule to string!")
        # torchIRstring = torchIR.operation.get_asm()
        torchIR.operation.write_bytecode(self.ir_path)
        # import sys
        # print(sys.getsizeof(str(torchIR)))
        # print(sys.getsizeof(torchIRstring))
        print("TorchIRModule converted to string!")
        del torchIR
        gc.collect()
        # def walk(op):
        #     print(str(op.get_asm()))
        #     if op.operation.name == "torch.vtensor.literal":
        #         for at in op.operation.attributes:
        #             print(at)
        #         attr = op.operation.attributes["value"]
        #         match = re.search(r'dense_resource<([^>]+)>\s*:\s*tensor<(\d+)x(\w+)>', str(attr))
        #         resource = match.group(1)
        #         shape = match.group(2)
        #         dtype = match.group(3)
        #         shape = [int(x) for x in shape.split("x")]
        #         attr = DenseResourceElementsAttr(attr)
        #         print(dir(attr))
        #         # print(op.operation.attributes[str(attr)])
        #         # x = torch.zeros(*shape)
        #         # if isinstance(attr, DenseResourceElementsAttr):
        #         #     print(dir(attr))
        #         #     attr.get_from_buffer(x, resource)
        #         # print(x)
        #     # for name in op.attributes:
        #     #     if(hasattr(name.attr, 'value')):
        #     #         print(name.attr.value)
            
        #     return WalkResult.ADVANCE
        # torchIR.operation.walk(walk)
        # del torchIR
        print("Starting diffusion graph compilation!")
        IRDict = self.compiler.compile(self.ir_path)
        print("Completed graph compilation!")
        return IRDict
