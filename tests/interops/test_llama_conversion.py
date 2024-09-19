import torch
from unittest import TestCase
from fmengine.models.llama.interop_llama import to_huggingface, from_huggingface
import transformers


class TestLlamaConversion(TestCase):
    def test_conversion(self, model_name="TinyLlama/TinyLlama_v1.1"):

        hf_reference = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

        from_hf_model, config = from_huggingface(model_name, "bfloat16", None)
        to_hf_model, _ = to_huggingface({"model": from_hf_model.state_dict()}, "bfloat16", config)

        to_hf_model = to_hf_model.state_dict()
        hf_reference = hf_reference.state_dict()

        for k in hf_reference:
            self.assertTrue(torch.allclose(hf_reference[k], to_hf_model[k]))
