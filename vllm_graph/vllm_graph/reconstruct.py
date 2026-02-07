from torch.fx import Graph
from torch.export.graph_signature import InputKind
from compiler import GraphCompiler
from vllm_graph.modelmaps import TYPE_MAP, OP_MAP
from vllm_graph.utils import read_pb
import torch
import json
import os
from collections import deque

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
    def __init__(self, graph_dict: dict, weight_path: str, arg_dict: dict):
        super().__init__()
        self.weights = read_pb(weight_path)
        self.graph_dict = graph_dict
        for buffer in arg_dict:
            if arg_dict[buffer]["kind"] == "buffer":
                self.register_buffer(arg_dict[buffer]["target"], arg_dict[buffer]["value"], persistent = True)

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

            data_name = f"weight_datasets{constant}" if self.graph_dict[constant].get("resource", None) is None else self.graph_dict[constant]["resource"]
            data = self.weights[data_name]
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
            del data
            setattr(self, var_name, weight)

    @property
    def device(self):
        """Returns the device of the model by checking the first parameter"""
        return next(self.parameters()).device

class vLLMGraphModel(torch.nn.Module):
    def __init__(self, graph_modules: dict[str, torch.fx.GraphModule], weights_directory: str):
        super().__init__()

        if graph_modules.get("main", None) is None:
            raise ValueError("FATAL: Model has no main module!")
        
        for func_name, graph_module in graph_modules.items():
            setattr(self, func_name, graph_module)
        
        self.weights = read_pb(weights_directory)
        vocab_size = self.main.config.vocab_size
        embedding_size = self.main.config.embedding_size
        weight_name = f"torch_tensor_{vocab_size}_{embedding_size}_torch.float32"
        data = self.weights[weight_name]
        tensor = torch.tensor(data, dtype = torch.float32)
        tensor = torch.reshape(tensor, (vocab_size, embedding_size))
        self.embeddings = torch.nn.Parameter(tensor, requires_grad=False)
        self.func_names = list(graph_modules.keys())
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = torch.device(args[0])
        for func_name in self.func_names:
            module = getattr(self, func_name)
            module = module.to(self.device)
            module.device = self.device
            
        return self
        
    def forward(self,
                input_ids,
                positions,
                intermediate_tensors=None,
                inputs_embeds=None,
                ):
        return self.main(input_ids, positions)

class vLLMGraph:
    def __init__(self, model_name: str, temp_directory: str | None = None, debug: bool = False):
        self.name = model_name
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
    
    def compile(self, model: torch.nn.Module, inputs: torch.Tensor, input_kwargs : dict = {}, dynamic_dims = {}):
        """
        Compiles the model and returns a topologically unsorted
        graph in dictionary format and stores it in graph_dict object of the class 
        """
        
        dynamo_model = torch.export.export(model, inputs, input_kwargs, dynamic_shapes = dynamic_dims)
        #TODO: Figure out a way to calculate this
        if hasattr(model, "config"):
            self.config = model.config
        else:
            self.config = None

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
        self.graph_dict["weights_directory"] = self.weights_directory
        self.graph_dict["model_name"] = self.name
        self.graph_dict["config"] = self.config
    
    def store_graph_dict(self):
        """Stores the graph dict for debugging purposes"""

        assert len(self.graph_dict) != 0, "Model not compiled"
        with open(f"{self.temp_directory}/model.json",'w') as f:
            f.write(json.dumps(self.graph_dict, indent = 2))
        
        print(f"model.json stored in {self.temp_directory}")
    
    def get_graph_dict(self):
        assert len(self.graph_dict) != 0, "Model not compiled"

        return ModelDict(self.graph_dict)


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
            kwarg_id = [ "scale", "enable_gqa"]
            i = 0
            for inp in graph_dict[node]['input_nodes'][:6]:
                input_args.append(graph_nodes[inp])
            
            for inp in graph_dict[node]['input_nodes'][6:]:
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
        
        elif node_type == "vllm_graph.vllm.arange":
            arange_func = OP_MAP.get(node_type, None)
            input_args = []
            for inp in graph_dict[node]['input_nodes']:
                input_args.append(graph_nodes[inp])
            
            input_kwargs = { "device" : graph.get_attr("device")}

            graph_nodes[node] = graph.call_function(arange_func, args=tuple(input_args), kwargs = input_kwargs)

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

def reconstruct_model(graph_dict: ModelDict) -> dict[str, torch.fx.GraphModule]:

    assert len(graph_dict) != 0, "Model not compiled"
    weights_directory = graph_dict["weights_directory"]

    graph_models = {}
    for func_name in graph_dict:
        if func_name in ['weights_directory', 'config', 'model_name']:
            continue
        
        arg_dict = graph_dict[func_name].get("arg_dict", {})
        model = ParameterModel(graph_dict[func_name], weights_directory, arg_dict)
        Nodes = []
        for node in graph_dict[func_name]:
            if node in ['entrypoint', 
                        'constants', 
                        'results', 
                        'weights_directory', 
                        'arg_dict',
                        'model_name',
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





        
            
        





