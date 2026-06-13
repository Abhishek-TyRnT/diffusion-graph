from torch.fx import Graph
from torch.export.graph_signature import InputKind
from compiler import GraphCompiler
import torch
import numpy as np

import os
import json 

class DiffusionGraphCompiler:
    def __init__(self, model_name: str, temp_directory: str | None = None, debug: bool = False):
        self.name = model_name
        if temp_directory is None:
            root_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.temp_directory = f"{root_folder}/temp_files/{model_name}"
            self.weights_directory = f"{root_folder}/temp_files/{model_name}"
        else:
            self.temp_directory = temp_directory
            self.weights_directory = f"{temp_directory}/{model_name}"
        
        if not os.path.exists(self.weights_directory):
            os.makedirs(self.weights_directory)
        self.graph_compiler = GraphCompiler(self.weights_directory, debug = debug)
        self.graph_dict : dict = {}
    
    def compile(self, model: torch.nn.Module, inputs: torch.Tensor, input_kwargs : dict = {}, dynamic_dims = {}):
        """
        Compiles the model and returns a topologically unsorted
        graph in dictionary format and stores it in graph_dict object of the class 
        """
        
        dynamo_model = torch.export.export(model, inputs, input_kwargs, dynamic_shapes = dynamic_dims)

        print("Completed Dynamo Export!")
        #TODO: Figure out a way to calculate this
        # if hasattr(model, "config"):
        #     self.config = model.config
        # else:
        self.config = None

        graph_signature = dynamo_model.graph_signature
        input_specs = graph_signature.input_specs
        index = 0
        self.arg_dict = {}
        print("Processing Input Specs!")
        for spec in input_specs:
            kind = spec.kind
            #Buffers
            if kind == InputKind.BUFFER:
                sub_model = model
                for target in spec.target.split("."):
                    sub_model = getattr(sub_model, target)
                self.arg_dict[index] = {"target" : spec.target.replace(".", "_"),
                                        "kind": "buffer",
                                        "value": sub_model
                                        }
                index +=1
            
            #user_inputs
            elif kind == InputKind.USER_INPUT:
                self.arg_dict[index] =  {"target" : spec.target, "kind": "user_input"}
                index += 1

        self.graph_dict = self.graph_compiler.compile(dynamo_model, inputs)
        print("Completed Graph Compilation!")
        input_args = self.graph_dict['main']["entrypoint"]
        new_args = []
        for arg in input_args:
            arg_index = int(arg.split("g")[-1])
            if self.arg_dict[arg_index]["kind"] == "buffer":
                next_nodes = self.graph_dict['main'][arg]["next_nodes"]
                for node in next_nodes:
                    input_nodes = self.graph_dict['main'][node]["input_nodes"]
                    index = input_nodes.index(arg)
                    self.graph_dict['main'][node]["input_nodes"][index] = self.arg_dict[arg_index]["target"]
                
                del self.graph_dict['main'][arg]
        
            else:
                new_args.append(arg)

        self.graph_dict["main"]["entrypoint"] = new_args
        self.graph_dict["main"]["arg_dict"] = self.arg_dict
        # self.graph_dict["weights_directory"] = self.weights_directory
        self.graph_dict["model_name"] = self.name
        self.graph_dict["config"] = self.config
    
    def store_graph_dict(self):
        """Stores the graph dict for debugging purposes"""

        assert len(self.graph_dict) != 0, "Model not compiled"
        buffers = {}
        path = "buffers.npz"
        for key in self.arg_dict:
            if self.arg_dict[key]["kind"] == "buffer":
                value = self.arg_dict[key]["value"]
                buffers[self.arg_dict[key]["target"]] = value.cpu().numpy()
                self.graph_dict["main"]["arg_dict"][key]["value"] = path

        if(len(buffers) != 0):
            np.savez(f"{self.weights_directory}/buffers.npz", **buffers)
    
        with open(f"{self.weights_directory}/model.json",'w') as f:
            f.write(json.dumps(self.graph_dict, indent = 2))
        
        print(f"model.json stored in {self.weights_directory}")
    
    def get_graph_dict(self):
        assert len(self.graph_dict) != 0, "Model not compiled"

        return self.graph_dict
