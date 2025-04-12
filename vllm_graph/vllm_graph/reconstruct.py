from torch.fx import Graph
from torch.export.graph_signature import InputKind
from compiler import GraphCompiler
from vllm_graph.modelmaps import TYPE_MAP, OP_MAP
from vllm_graph.utils import read_pb
import torch
import json
import os
from collections import deque


class vLLMGraphModel(torch.nn.Module):
    def __init__(self, graph_dict: dict, weight_path: str, arg_dict: dict):
        super().__init__()
        self.weights = read_pb(weight_path)
        self.graph_dict = graph_dict
        for buffer in arg_dict:
            if arg_dict[buffer]["kind"] == "buffer":
                self.register_buffer(arg_dict[buffer]["target"], arg_dict[buffer]["value"], persistent = True)

        for constant in self.graph_dict["constants"]:
            data_name = f"weight_datasets{constant}"
            data = self.weights[data_name]
            dtype = self.graph_dict[constant]['dtype']

            if(self.graph_dict[constant]["vllm_graph_type"] == "tuple"):
                ssa_id = constant.split(".")[0]
                var_name = f"weight_{ssa_id}"
                setattr(self, var_name, tuple(data))
                continue

            if(self.graph_dict[constant].get("output_shape", None) is None):
                ssa_id = constant.split(".")[0]
                var_name = f"weight_{ssa_id}"
                setattr(self, var_name, data)
                continue
            
            
            tensor = torch.tensor(data, dtype = TYPE_MAP[dtype])
            tensor = torch.reshape(tensor, self.graph_dict[constant]["output_shape"])
            
            ssa_id = constant.split(".")[0]
            var_name = f"weight_{ssa_id}"
            weight = torch.nn.Parameter(tensor, requires_grad=False)
            del data
            setattr(self, var_name, weight)


class vLLMGraph:
    def __init__(self, model_name: str, temp_directory: str | None = None, debug: bool = False):
        if temp_directory is None:
            root_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.temp_directory = f"{root_folder}/temp_files/{model_name}"
            self.weights_directory = f"{root_folder}/temp_files/{model_name}/weights.pb"
        else:
            self.temp_directory = f"{temp_directory}/{model_name}"
            self.weights_directory = f"{temp_directory}/{model_name}/weights.pb"
        
        if not os.path.exists(self.temp_directory):
            os.makedirs(self.temp_directory)
        self.graph_compiler = GraphCompiler(self.weights_directory, debug = debug)
        self.graph_dict : dict = {}
    
    def compile(self, model: torch.nn.Module, inputs: torch.Tensor, input_kwargs : dict = {}):
        """
        Compiles the model and returns a topologically unsorted
        graph in dictionary format and stores it in graph_dict object of the class 
        """
        
        dynamo_model = torch.export.export(model, inputs, input_kwargs, dynamic_shapes = None)
        graph_signature = dynamo_model.graph_signature
        input_specs = graph_signature.input_specs
        index = 0
        self.arg_dict = {}
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
        input_args = self.graph_dict["entrypoint"]
        new_args = []
        for arg in input_args:
            arg_index = int(arg.split("g")[-1])
            if self.arg_dict[arg_index]["kind"] == "buffer":
                next_nodes = self.graph_dict[arg]["next_nodes"]
                for node in next_nodes:
                    input_nodes = self.graph_dict[node]["input_nodes"]
                    index = input_nodes.index(arg)
                    self.graph_dict[node]["input_nodes"][index] = self.arg_dict[arg_index]["target"]
                
                del self.graph_dict[arg]
        
            else:
                new_args.append(arg)
        self.graph_dict["entrypoint"] = new_args
    
    def store_graph_dict(self):
        """Stores the graph dict for debugging purposes"""
    
        assert len(self.graph_dict) != 0, "Model not compiled"
        with open(f"{self.temp_directory}/model.json",'w') as f:
            f.write(json.dumps(self.graph_dict, indent = 2))
        
        print(f"model.json stored in {self.temp_directory}")
    
    def get_graph_dict(self):
        assert len(self.graph_dict) != 0, "Model not compiled"

        return self.graph_dict

    def topological_sort(self, graph_dict: dict, Nodes: list)->list:
        # Vector to store indegree of each vertex
        indegree = {vertex: 0 for vertex in Nodes}
        for ssa_id in Nodes:
            for vertex in graph_dict[ssa_id].get("next_nodes", []):
                indegree[vertex] += 1

        # Queue to store vertices with indegree 0
        q = deque()
        for ssa_id in Nodes:
            if indegree[ssa_id] == 0:
                q.append(ssa_id)
        result = []
        while q:
            node = q.popleft()
            result.append(node)
            # Decrease indegree of adjacent vertices as the current node is in topological order
            for adjacent in graph_dict[node].get("next_nodes", []):
                indegree[adjacent] -= 1
                # If indegree becomes 0, push it to the queue
                if indegree[adjacent] == 0:
                    q.append(adjacent)

        # Check for cycle
        if len(result) != len(Nodes):
            raise ValueError("InValid Graph!, Graph is not topologically sortable")
        
        return result

    def construct_graph(self, nodes: list[str], results: list[str]) -> torch.fx.Graph:
        """Reconstructs torch.fx.Graph from grapg dict."""
        graph_nodes = {}
        graph = torch.fx.Graph()
        for buffer_args in self.arg_dict:
            if self.arg_dict[buffer_args]['kind'] == "buffer":
                target = self.arg_dict[buffer_args]['target']
                graph_nodes[target] = graph.get_attr(target)
        
        
        for node in nodes:
            node_type = self.graph_dict[node].get("op_name", None)
            if(node_type is None):
                raise ValueError(f"op name for {node} is invalid")
            
            if node_type == "input_arg":
                graph_nodes[node] = graph.placeholder(node)
            
            elif node_type == "arith.constant" or \
                 node_type == "vllm_graph.vllm.const_tuple" or \
                 node_type == "vllm_graph.constant.tensor" :
                ssa_id = node.split(".")[0]
                graph_nodes[node] = graph.get_attr(f"weight_{ssa_id}")
            
            elif node_type == "vllm_graph.vllm.list_op":
                ssa_id = node.split(".")[0]
                list_nodes = [graph_nodes[inp] for inp in self.graph_dict[node]['input_nodes']]
                graph_nodes[node] = list_nodes

            elif node_type in ["vllm_graph.vllm.add", "vllm_graph.vllm.sub"]:
                add_func = OP_MAP.get(node_type, None)
                input_args = []
                input_kwargs = {}

                #TODO: Find a better way deal with kwargs
                for inp in self.graph_dict[node]['input_nodes'][:-1]:
                    input_args.append(graph_nodes[inp])
                
                inp_alpha = self.graph_dict[node]['input_nodes'][-1]
                input_kwargs["alpha"] = graph_nodes[inp_alpha]
                
                graph_nodes[node] = graph.call_function(add_func, args=tuple(input_args), kwargs = input_kwargs)

            elif node_type == "vllm_graph.vllm.addmm":
                add_func = OP_MAP.get(node_type, None)
                input_args = []
                input_kwargs = {}
                for inp in self.graph_dict[node]['input_nodes']:
                    if self.graph_dict[inp]["vllm_graph_type"] == "scalar" and input_kwargs.get("alpha", None) is None:
                        input_kwargs["alpha"] = graph_nodes[inp]
                    
                    elif self.graph_dict[inp]["vllm_graph_type"] == "scalar":
                        input_kwargs["beta"] = graph_nodes[inp]
                    else:
                        input_args.append(graph_nodes[inp])
                
                graph_nodes[node] = graph.call_function(add_func, args=tuple(input_args), kwargs = input_kwargs)
                
            else:
                op_func = OP_MAP.get(node_type, None)
                if op_func is None:
                  raise ValueError(f"op function for {node_type} is not defined!")
                input_args = []
                for inp in self.graph_dict[node]['input_nodes']:
                    input_args.append(graph_nodes[inp])
                
                graph_nodes[node] = graph.call_function(op_func, args=tuple(input_args))
        
        result_nodes = []
        for result_ssa_id in results:
            result_nodes.append(graph_nodes[result_ssa_id])
        graph.output(result_nodes)
        return graph
                


    def reconstruct(self) -> torch.fx.GraphModule:

        assert len(self.graph_dict) != 0, "Model not compiled"
        model = vLLMGraphModel(self.graph_dict, self.weights_directory, self.arg_dict)
        Nodes = []
        for node in self.graph_dict:
            if node in ['entrypoint', 'constants', 'results']:
                continue
            Nodes.append(node)
        
        #Topologically sorted nodes will ensure we don't have node as input which was
        #not declared before.
        topologically_sorted_nodes = self.topological_sort(self.graph_dict, Nodes)
        
        #Topological sort changes of the order of the arguments, which can lead to
        #unpredictable output
        argless_topologically_sorted_nodes = []
        for node in topologically_sorted_nodes:
            if "arg" in node:
                continue
            argless_topologically_sorted_nodes.append(node)
        
        topologically_sorted_nodes = self.graph_dict["entrypoint"] + argless_topologically_sorted_nodes
        
        module_graph = self.construct_graph(topologically_sorted_nodes, self.graph_dict["results"])
        return torch.fx.GraphModule(model, module_graph)





        
            
        





