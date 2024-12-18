from graph_compiler import vLLMGraph
import torch
from torch.nn import Module
from torch_mlir import torchscript


BACKEND_END_LEGAL_OPS = ["aten.softmax.int"]
class GraphCompiler:
    def __init__(self, weight_path: str, debug: bool = False):
        self.compiler = vLLMGraph(weight_path)
        self.debug = debug
        self.backend_legal_ops = BACKEND_END_LEGAL_OPS

    def compile(self, model: Module, inputs: list[torch.Tensor]) -> dict:
        torchIR = torchscript.compile(model, inputs, output_type="torch", backend_legal_ops = self.backend_legal_ops)
        IRDict = self.compiler.compile(str(torchIR))
        return IRDict
