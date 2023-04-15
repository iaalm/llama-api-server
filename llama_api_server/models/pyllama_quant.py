import json
from pathlib import Path
from llama_api_server.utils import get_uuid, get_timestamp


class PyLlamaQuantCompletion:
    def __init__(self, params):
        try:
            import llama
            import torch
            from llama.hf import LLaMATokenizer
            from llama.hf.utils import get_llama
            from llama.llama_quant import load_quant
        except ImportError:
            raise ImportError(
                'To run model with pyllama, please run "python -m pip install pyllama transformers" first'
            )
        max_seq_len = params.get("max_seq_len", None) or 2048
        max_batch_size = params.get("max_batch_size", None) or 16
        wbits = params.get("wbits", None) or 4
        path = params["path"]
        device = params.get("device", "cuda")
        model_name = params.get("model_name", "decapoda-research/llama-7b-hf")
        self.model = load_quant(model_name, path, wbits, max_seq_len)
        self.dev = torch.device(device)

        self.model.to(self.dev)
        self.tokenizer = LLaMATokenizer.from_pretrained(model_name)

    def completions(self, args):
        import torch

        prompt = args["prompt"]
        top_p = args["top_p"]
        suffix = args["suffix"]
        echo = args["echo"]
        temp = args["temperature"]
        max_tokens = args["max_tokens"]

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.dev)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                min_length=1,
                max_new_tokens=max_tokens,
                top_p=top_p,
                temperature=temp,
            )
        result = self.tokenizer.decode([el.item() for el in generated_ids[0]])
        finish_reason = "length"
        c_prompt_tokens = len(input_ids)
        c_completion_tokens = len(generated_ids)
        return {
            "id": get_uuid(),
            "object": "text_completion",
            "created": get_timestamp(),
            "model": args["model"],
            "choices": [
                {
                    "text": result + suffix,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": c_prompt_tokens,
                "completion_tokens": c_completion_tokens,
                "total_tokens": c_prompt_tokens + c_completion_tokens,
            },
        }
