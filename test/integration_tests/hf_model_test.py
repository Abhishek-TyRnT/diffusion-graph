import pytest
import json
import torch
from vllm_graph.reconstruct import vLLMGraph
from transformers import AutoTokenizer, AlbertModel

@pytest.mark.parametrize("model_name, text, model_class",
    (["albert/albert-base-v2", "Hello, my dog is cute", AlbertModel],))
def test_hf_models(model_name, text, model_class):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, attn_implementation = None)
    
    inputs = tokenizer(text, return_tensors="pt")
    model(**inputs)
    tmp_folder = f"/tmp/{model.__class__.__name__}"

    vllmgraph = vLLMGraph(tmp_folder)
    vllmgraph.compile(model, list(inputs.values()))
