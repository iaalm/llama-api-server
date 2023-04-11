class PyLlamaCompletion:
    def __init__(self, params):
        try:
            import llama
        except ImportError:
            raise ImportError(
                'To run model with pyllama, please run "python -m pip install pyllama transformers" first'
            )
        local_rank = 0
        world_size = 1
        max_seq_len = params.get("max_seq_len", None) or 2048
        max_batch_size = params.get("max_batch_size", None) or 16
        ckpt_dir = params["ckpt_dir"]
        tokenizer_path = params["tokenizer_path"]
        self.model = load(
            ckpt_dir,
            tokenizer_path,
            local_rank,
            world_size,
            max_seq_len,
            max_batch_size,
        )

    def completions(self, args):
        prompt = args["prompt"]
        top_p = args["top_p"]
        suffix = args["suffix"]
        echo = args["echo"]
        temp = args["temperature"]
        max_tokens = args["max_tokens"]
        result = prompt if echo else ""
        result += generator.generate(
            [prompt], max_gen_len=max_gen_len, temperature=temp, top_p=top_p
        )
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
