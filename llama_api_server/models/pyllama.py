import os
import json
from pathlib import Path
from llama_api_server.utils import get_uuid, get_timestamp


class PyLlamaCompletion:
    def __init__(self, params):
        try:
            import llama
            import torch
        except ImportError:
            raise ImportError(
                'To run model with pyllama, please run "python -m pip install pyllama transformers" first'
            )
        local_rank = 0
        world_size = 1
        max_seq_len = params.get("max_seq_len", None) or 2048
        max_batch_size = params.get("max_batch_size", None) or 2
        ckpt_dir = params["ckpt_dir"]
        tokenizer_path = params["tokenizer_path"]
        device = params.get("device", "cuda")
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert world_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
        ckpt_path = checkpoints[local_rank]

        checkpoint = torch.load(ckpt_path, map_location="cpu")

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = llama.ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        )
        tokenizer = llama.Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        if device.startswith("cuda"):
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            os.environ['KV_CAHCHE_IN_GPU'] = "0"
            torch.set_default_tensor_type(torch.FloatTensor)
        model = llama.Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False)
        self.model = llama.LLaMA(model, tokenizer)

    def completions(self, args):
        prompt = args["prompt"]
        top_p = args["top_p"]
        suffix = args["suffix"]
        echo = args["echo"]
        temp = args["temperature"]
        max_tokens = args["max_tokens"]
        result = self.model.generate(
            [prompt], max_gen_len=max_tokens, temperature=temp, top_p=top_p
        )[0]
        finish_reason = "length"
        c_prompt_tokens = n_past = 0
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
                "completion_tokens": n_past - c_prompt_tokens,
                "total_tokens": n_past,
            },
        }


