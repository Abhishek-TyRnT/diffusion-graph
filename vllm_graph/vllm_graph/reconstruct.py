from torch.fx import Graph
from torch.export.graph_signature import InputKind
from compiler import GraphCompiler
from vllm_graph.modelmaps import TYPE_MAP, OP_MAP
from vllm_graph.utils import read_pb
import torch
import numpy as np
import json
import os
from collections import deque
import gc

class ModelDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = args[0]["config"]
        self.name = args[0]["model_name"]
    def __str__(self):
        return f"<{self.name}>"
    
    def __repr__(self):
        return f"<{self.name}>"

class ParameterModel(torch.nn.Module):
    def __init__(self, graph_dict: dict, weightPathMap: dict, arg_dict: dict, weights_directory: str):
        super().__init__()
        self.weight_dict = {"DenseElementsAndScalars" : read_pb(f"{weights_directory}/{weightPathMap['DenseAndScalarWeights']}")}
        self.graph_dict = graph_dict
        buffer_dict = None
        for buffer in arg_dict:
            if arg_dict[buffer]["kind"] == "buffer":
                #This is done to store register buffers during compilation
                if(isinstance(arg_dict[buffer]["value"], str)):
                    if(buffer_dict is None):
                        buffer_dict = np.load(arg_dict[buffer]["value"])
                    
                    value = buffer_dict[arg_dict[buffer]["target"]]
                    value = torch.tensor(value)
                else:
                    value = arg_dict[buffer]["value"]
                self.register_buffer(arg_dict[buffer]["target"], value, persistent = True)

        for constant in self.graph_dict["constants"]:
            if(self.graph_dict[constant]["dtype"] == "!vllm_graph.none"):
                ssa_id = constant.replace(".","_")
                var_name = f"weight_{ssa_id}"
                setattr(self, var_name, None)
                continue

            if(self.graph_dict[constant]["dtype"] == "!vllm_graph.str"):
                ssa_id = constant.replace(".","_")
                var_name = f"weight_{ssa_id}"
                setattr(self, var_name, self.graph_dict[constant]["value"])
                continue

            if self.graph_dict[constant].get("resource", None) is None:
                data_name = f"weight_datasets{constant}"
                data = self.weight_dict["DenseElementsAndScalars"][data_name]
                key = "DenseElementsAndScalars"
                
            else:
                data_name = self.graph_dict[constant]["resource"]
                if weightPathMap[data_name] not in self.weight_dict:
                    self.weight_dict[weightPathMap[data_name]] = read_pb(f"{weights_directory}/{weightPathMap[data_name]}")

                data = self.weight_dict[weightPathMap[data_name]][data_name]
                key = weightPathMap[data_name]
            
            dtype = self.graph_dict[constant]['dtype']

            if(self.graph_dict[constant]["vllm_graph_type"] == "tuple"):
                ssa_id = constant.replace(".","_")
                var_name = f"weight_{ssa_id}"
                setattr(self, var_name, tuple(data))
                continue

            if(self.graph_dict[constant].get("output_shape", None) is None):
                ssa_id = constant.replace(".","_")
                var_name = f"weight_{ssa_id}"
                if(dtype == "i1"):
                    data = bool(data)
                setattr(self, var_name, data)
                continue
            
            
            tensor = torch.tensor(data, dtype = TYPE_MAP[dtype])
            tensor = torch.reshape(tensor, self.graph_dict[constant]["output_shape"])
            
            ssa_id = constant.replace(".","_")
            var_name = f"weight_{ssa_id}"
            weight = torch.nn.Parameter(tensor, requires_grad=False)
            del self.weight_dict[key][data_name]
            setattr(self, var_name, weight)

        del self.weight_dict
        gc.collect()

    @property
    def device(self):
        """Returns the device of the model by checking the first parameter"""
        return next(self.parameters()).device

def construct_graph(graph_dict: dict, arg_dict: dict, nodes: list[str], results: list[str]) -> torch.fx.Graph:
    """Reconstructs torch.fx.Graph from grapg dict."""
    graph_nodes = {}
    graph = torch.fx.Graph()
    for buffer_args in arg_dict:
        if arg_dict[buffer_args]['kind'] == "buffer":
            target = arg_dict[buffer_args]['target']
            graph_nodes[target] = graph.get_attr(target)
    
    for node in nodes:
        node_type = graph_dict[node].get("op_name", None)
        if(node_type is None):
            raise ValueError(f"op name for {node} is invalid")
        
        if node_type != "input_arg":
            node_category = node_type.split(".")[1]

            if(node_category == "temp"):
                raise NotImplementedError(f"Temporary op {node_type} not supported!")
        
        if node_type == "input_arg":
            graph_nodes[node] = graph.placeholder(node)
        
        elif node_type == "arith.constant" or \
                node_type == "vllm_graph.vllm.const_tuple" or \
                node_type == "vllm_graph.constant.tensor" or \
                node_type == "vllm_graph.constant.none" or \
                node_type == "vllm_graph.constant.string":
            ssa_id = node.replace(".","_")
            graph_nodes[node] = graph.get_attr(f"weight_{ssa_id}")
        
        elif node_type == "vllm_graph.vllm.list_op":
            ssa_id = node.split(".")[0]
            list_nodes = [graph_nodes[inp] for inp in graph_dict[node]['input_nodes']]
            graph_nodes[node] = list_nodes

        elif node_type in ["vllm_graph.vllm.add", "vllm_graph.vllm.sub"]:
            add_func = OP_MAP.get(node_type, None)
            input_args = []
            input_kwargs = {}

            #TODO: Find a better way deal with kwargs
            for inp in graph_dict[node]['input_nodes'][:-1]:
                input_args.append(graph_nodes[inp])
            
            inp_alpha = graph_dict[node]['input_nodes'][-1]
            input_kwargs["alpha"] = graph_nodes[inp_alpha]
            
            graph_nodes[node] = graph.call_function(add_func, args=tuple(input_args), kwargs = input_kwargs)

        elif node_type == "vllm_graph.vllm.addmm":
            add_func = OP_MAP.get(node_type, None)
            input_args = []
            input_kwargs = {}
            for inp in graph_dict[node]['input_nodes']:
                if graph_dict[inp]["vllm_graph_type"] == "scalar" and input_kwargs.get("alpha", None) is None:
                    input_kwargs["alpha"] = graph_nodes[inp]
                
                elif graph_dict[inp]["vllm_graph_type"] == "scalar":
                    input_kwargs["beta"] = graph_nodes[inp]
                else:
                    input_args.append(graph_nodes[inp])
            
            graph_nodes[node] = graph.call_function(add_func, args=tuple(input_args), kwargs = input_kwargs)
        elif node_type == "vllm_graph.vllm.scaled_dot_product_attention":
            attn_func = OP_MAP.get(node_type, None)
            input_args = []
            input_kwargs = {}
            kwarg_id = [ "scale"]
            i = 0
            for inp in graph_dict[node]['input_nodes'][:6]:
                input_args.append(graph_nodes[inp])
            
            for inp in graph_dict[node]['input_nodes'][6:7]:
                input_kwargs[kwarg_id[i]] = graph_nodes[inp]
                i+=1
            graph_nodes[node] = graph.call_function(attn_func, args=tuple(input_args), kwargs = input_kwargs)

        elif node_type == "vllm_graph.vllm.gelu":
            gelu_func = OP_MAP.get(node_type, None)
            input_args = []
            for inp in graph_dict[node]['input_nodes'][:1]:
                input_args.append(graph_nodes[inp])
            
            input_kwargs = {"approximate":graph_nodes[graph_dict[node]['input_nodes'][1]]}
            graph_nodes[node] = graph.call_function(gelu_func, args=tuple(input_args), kwargs = input_kwargs)
            
        elif node_type == "vllm_graph.vllm.ones":
            ones_func = OP_MAP.get(node_type, None)
            i = 0
            input_kwargs = {}
            input_args = []
            kwarg_id = [ "dtype", "layout"]
            for inp in graph_dict[node]['input_nodes'][:1]:
                input_args.append(graph_nodes[inp])

            for inp in graph_dict[node]['input_nodes'][1:]:
                input_kwargs[kwarg_id[i]] = graph_nodes[inp]
                i+=1
            input_kwargs['device'] = graph.get_attr("device")
            graph_nodes[node] = graph.call_function(ones_func, args=tuple(input_args), kwargs = input_kwargs)
                
        elif node_type == "vllm_graph.vllm.upsample":
            upsample_func = OP_MAP.get(node_type, None)
            i = 0
            input_kwargs = {}
            input_args = []
            kwarg_id = ["scale_factor","mode"]
            for inp in graph_dict[node]['input_nodes'][:1]:
                input_args.append(graph_nodes[inp])

            for inp in graph_dict[node]['input_nodes'][1:]:
                input_kwargs[kwarg_id[i]] = graph_nodes[inp]
                i+=1

            graph_nodes[node] = graph.call_function(upsample_func, args=tuple(input_args), kwargs = input_kwargs)


        else:
            op_func = OP_MAP.get(node_type, None)
            if op_func is None:
                raise ValueError(f"op function for {node_type} is not defined!")
            input_args = []
            for inp in graph_dict[node]['input_nodes']:
                input_args.append(graph_nodes[inp])
            
            graph_nodes[node] = graph.call_function(op_func, args=tuple(input_args))
        
        #Keep this code snippet for future debugging
        # func = lambda name, exp_shape, x : print(f"Op Name :- {name}, Expected Shape :- {exp_shape}, Actual Shape :- {x.shape if hasattr(x, 'shape') else x}")
        # graph.call_function(
        #             func,
        #             args=(node, graph_dict[node].get("output_shape", None), graph_nodes[node])
        #         )
        
    
    if len(results) == 1:
        graph.output(graph_nodes[results[0]])
    else:
        result_nodes = []
        for result_ssa_id in results:
            result_nodes.append(graph_nodes[result_ssa_id])
        graph.output(result_nodes)
    return graph


def topological_sort(graph_dict: dict, Nodes: list)->list:
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

def reconstruct_model(graph_dict: ModelDict, weights_directory: str) -> dict[str, torch.fx.GraphModule]:

    assert len(graph_dict) != 0, "Model not compiled"
    weightPathMap = graph_dict.get("resources_path", {})
    weightPathMap.update({"DenseAndScalarWeights" : graph_dict["DenseAndScalarWeights"]})

    graph_models = {}
    for func_name in graph_dict:
        if func_name in ['resources_path', 'DenseAndScalarWeights', 'config', 'model_name']:
            continue
        
        arg_dict = graph_dict[func_name].get("arg_dict", {})
        model = ParameterModel(graph_dict[func_name], weightPathMap, arg_dict, weights_directory)
        Nodes = []
        for node in graph_dict[func_name]:
            if node in ['entrypoint', 
                        'constants', 
                        'results', 
                        'weights_directory', 
                        'arg_dict',
                        'model_name',
                        'DenseAndScalarWeights',
                        'resource_path',
                        'config']:
                continue
            Nodes.append(node)
        
        #Topologically sorted nodes will ensure we don't have node as input which was
        #not declared before.
        topologically_sorted_nodes = topological_sort(graph_dict[func_name], Nodes)
        
        #Topological sort changes of the order of the arguments, which can lead to
        #unpredictable output
        argless_topologically_sorted_nodes = []
        for node in topologically_sorted_nodes:
            if "arg" in node:
                continue
            argless_topologically_sorted_nodes.append(node)
        
        topologically_sorted_nodes = graph_dict[func_name]["entrypoint"] + argless_topologically_sorted_nodes
        module_graph = construct_graph(graph_dict[func_name], arg_dict, topologically_sorted_nodes, graph_dict[func_name]["results"])
        graph_model = torch.fx.GraphModule(model, module_graph)
        graph_model.config = graph_dict['config']
        graph_model.name = graph_dict['model_name']
        graph_models[func_name] = graph_model
    return graph_models





        
            
        





