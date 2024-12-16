from graph_compiler import vLLMGraph
import torch
from torch.nn import Module
from torch_mlir import torchscript


class GraphCompiler:
    def __init__(self, weight_path: str, debug: bool = False):
        self.compiler = vLLMGraph(weight_path)
        self.debug = debug

    def compile(self, model: Module, inputs: list[torch.Tensor]) -> dict:
        torchIR = torchscript.compile(model, inputs, output_type="torch")
        IRDict = self.compiler.compile(str(torchIR))
        return IRDict
