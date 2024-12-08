from torch.fx import Graph
from compiler import GraphCompiler
from vllm_graph.modelmaps import TYPE_MAP, OP_MAP
import torch
import json
import h5py
import os


class vLLMGraphModel(torch.nn.Module):
    def __init__(self, graph_dict: dict, weight_path: str):
        weights = h5py.File(weight_path, 'r')
        for constant in graph_dict["constants"]:
            data_name = f"weights_dataset{constant}"
            data = weights[data_name]
            dtype = graph_dict[constant]['dtype']
            tensor = torch.tensor(data, TYPE_MAP[dtype])
            if(graph_dict[constant].get("shape", None) is not None):
                tensor = torch.reshape(tensor, graph_dict[constant]["shape"])
            
            ssa_id = constant.split(".")[0]
            var_name = f"weight_{ssa_id}"
            weight = torch.nn.Parameter(tensor)
            setattr(self, var_name, weight)


class vLLMGraph:
    def __init__(self, model_name: str, temp_directory: str | None = None):
        if temp_directory is None:
            root_folder = os.path.dirname(os.path.dirname(__file__))
            self.temp_directory = f"{root_folder}/temp_files/{model_name}"
            self.weights_directory = f"{root_folder}/temp_files/{model_name}/weights.h5"
        else:
            self.temp_directory = f"{temp_directory}/{model_name}"
            self.weights_directory = f"{temp_directory}/{model_name}/weights.h5"
        
        self.graph_compiler = GraphCompiler(self.weights_directory)
        self.graph_dict : dict = {}
    
    def compile(self, model: torch.nn.Module):
        self.graph_dict = self.graph_compiler.compile(model)
        #This marks all the nodes as not visited
        #TODO: add this flag while creating dict
        for node in self.graph_dict:
            if node in ["entrypoint", "constants"]:
                continue
            self.graph_dict[node]['visited'] = False

    
    def store_graph_dict(self):
    
        assert len(self.graph_dict) != 0, "Model not compiled"
        with open(f"{self.temp_directory}/model.json") as f:
            f.write(json.dumps(self.graph_dict, indent = 2))
        
        print(f"model.json stored in {self.temp_directory}")
    

    
    def reconstruct(self) -> torch.nn.Module:
        node_graph = {}
        fx_graph = Graph()
        model = vLLMGraphModel(self.graph_dict, self.weights_directory)
        for arg in self.graph_dict['entrypoint']:
            node_graph[arg] = fx_graph.placeholder(arg)
        





