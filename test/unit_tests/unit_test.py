import pytest
import subprocess
import os

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
    print(cmd)
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    exit_code = process.returncode


    stdout = process.stdout.decode("utf-8")
    print(stdout)
    stderr = process.stderr.decode("utf-8")
    assert exit_code == 0, f"The test failed with response \n{stderr}"
    
