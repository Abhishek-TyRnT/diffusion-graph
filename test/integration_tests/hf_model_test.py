import pytest
import json
import torch
from vllm_graph.reconstruct import vLLMGraph
from transformers import AutoTokenizer, AlbertModel
from test_utils import validate_outputs

@pytest.mark.parametrize("model_name, text, model_class",
    (["albert/albert-base-v2", "Hello, my dog is cute", AlbertModel],))
def test_hf_models(model_name, text, model_class):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, attn_implementation = None)
    
    inputs = tokenizer(text, return_tensors="pt")
    model(**inputs)
    tmp_folder = f"./temp_files"

    vllmgraph = vLLMGraph(model_name,temp_directory = tmp_folder, debug = True)
    input_kwargs = {"return_dict" : False}
    vllmgraph.compile(model, tuple(inputs.values()), input_kwargs)
    reconstructed_model = vllmgraph.reconstruct()
    normal_output = model(*inputs)
    vllm_graph_output = reconstructed_model(*inputs)

    assert validate_outputs(vllm_graph_output, normal_output), f"Test failed validation check"

