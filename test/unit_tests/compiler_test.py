import pytest
import subprocess
import os
import sys
from torch_mlir import torchscript

def getmodel_path():
    model_path = os.path.dirname(__file__)
    model_path = os.path.dirname(model_path)
    return model_path

sys.path.append(getmodel_path())

from test_models import *

@pytest.mark.parametrize("filename",
    ["add_static_shapes.mlir",
     "add_dynamic_shapes.mlir",
     "Linear_static_shapes_with_bias.mlir",
     pytest.param("Linear_dynamic_shapes_with_bias.mlir", marks=pytest.mark.xfail),
     "Relu_static_shapes.mlir",
     "Relu_dynamic_shapes.mlir",
     "softmax.mlir",
     "transpose_dynamic_shapes.mlir",
     "transpose_static_shapes.mlir"
     ])
def test_vllm_graph_compiler_from_mlir(filename):
    #TODO: Add lit test for verification
    root_folder = os.path.dirname(__file__)
    root_folder = os.path.dirname(root_folder)

    cmd = ["vllm-graph" ,f"{root_folder}/examples/{filename}"]
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    exit_code = process.returncode


    stdout = process.stdout.decode("utf-8")
    print(stdout)
    stderr = process.stderr.decode("utf-8")
    assert exit_code == 0, f"The test failed with response \n{stderr}"
    
    
@pytest.mark.parametrize("model, pass_list, model_args, inputs",
    ([Add, ["convert-global-function-pass","func.func(convert-torch-to-vllm-graph)"],
            (), (torch.randn(224, 10, 3), torch.randn(224, 10, 3))],
     [LinearModule, ["convert-global-function-pass","func.func(convert-torch-to-vllm-graph, recompose-simple-ops-to-complex)"],
      (10, 5), (torch.randn(1, 224, 10),)],
     [ReluModule, ["convert-global-function-pass","func.func(convert-torch-to-vllm-graph)"], 
        (), (torch.randn(1, 10, 5))],
     [Softmax, ["convert-global-function-pass","func.func(convert-torch-to-vllm-graph)"], 
        (1,), (torch.randn(1, 10, 5))],
     [Transpose, ["convert-global-function-pass","func.func(convert-torch-to-vllm-graph)"], 
        (1, 0), (torch.randn(25, 10),)]
     ))
def test_vllm_graph_compiler_passes_from_models(model,
                                                pass_list,
                                                model_args,
                                                inputs):
    backend_legal_ops = ["aten.softmax.int"]
    if len(model_args) == 0:
        torch_model = model()
    else:
        torch_model = model(*model_args)
    
    torch_model.eval()
    torchIR = torchscript.compile(torch_model, inputs, output_type="torch", backend_legal_ops=backend_legal_ops)

    filename = f"/tmp/{torch_model.__class__.__name__}.mlir"
    with open(filename , "w") as f:
        f.write(str(torchIR))
    
    passes = ",".join(pass_list)
    
    cmd = f'vllm-graph-opt --pass-pipeline="builtin.module({passes})" --mlir-elide-elementsattrs-if-larger=20 {filename}'
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    exit_code = process.returncode


    stdout = process.stdout.decode("utf-8")
    print(stdout)
    stderr = process.stderr.decode("utf-8")
    assert exit_code == 0, f"The test failed with response \n{stderr}"


@pytest.mark.parametrize("model, model_args, inputs",
    ([Add, (), (torch.randn(224, 10, 3), torch.randn(224, 10, 3))],
     [LinearModule, (10, 5), (torch.randn(1, 224, 10),)],
     [ReluModule, (), (torch.randn(1, 10, 5))],
     [Softmax, (1,), (torch.randn(1, 10, 5))],
     [Transpose, (1, 0), (torch.randn(25, 10),)]
     ))
def test_vllm_graph_compiler_from_models(model,
                                        model_args,
                                        inputs):
    backend_legal_ops = ["aten.softmax.int"]
    if len(model_args) == 0:
        torch_model = model()
    else:
        torch_model = model(*model_args)
    
    torch_model.eval()
    torchIR = torchscript.compile(torch_model, inputs, output_type="torch", backend_legal_ops=backend_legal_ops)

    filename = f"/tmp/{torch_model.__class__.__name__}.mlir"
    with open(filename , "w") as f:
        f.write(str(torchIR))
    cmd = ["vllm-graph" ,filename]
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    exit_code = process.returncode


    stdout = process.stdout.decode("utf-8")
    print(stdout)
    stderr = process.stderr.decode("utf-8")
    assert exit_code == 0, f"The test failed with response \n{stderr}"