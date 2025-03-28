import pytest
import json
import torch
from vllm_graph.reconstruct import vLLMGraph
from transformers import AutoTokenizer, AlbertModel
from test_utils import validate_outputs
from transformers.models.albert.modeling_albert import AlbertEmbeddings, AlbertSdpaAttention, AlbertLayer, AlbertTransformer
from transformers import AlbertConfig

@pytest.mark.parametrize("Model, inputs, input_kwargs", (
    [AlbertEmbeddings, (torch.randint(0, 100, (8, 512)),), {}],
    [AlbertSdpaAttention, (torch.randn(8, 512, 4096),), {}],
    [AlbertLayer, (torch.randn(1, 8, 4096),), {}],
    [AlbertTransformer, (torch.randn(1, 8, 128),), {"return_dict": False}],
))
def test_hf_model_layer(Model,
                        inputs,
                        input_kwargs):
    config = AlbertConfig()
    model = Model(config)
    model.eval()
    tmp_folder = f"/tmp"

    vllmgraph = vLLMGraph(model.__class__.__name__, tmp_folder, )
    vllmgraph.compile(model, inputs, input_kwargs)
    reconstructed_model = vllmgraph.reconstruct()
    normal_output = model(*inputs)
    vllm_graph_output = reconstructed_model(*inputs)

    assert validate_outputs(vllm_graph_output, normal_output), f"Test failed validation check"


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

