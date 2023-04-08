import llamacpp
import array
from llama_api_server.utils import get_uuid, get_timestamp, unpack_cfloat_array


class LlamaCpp:
    def __init__(self, params):
        self.model_path = params["path"]
        self.n_batch = params.get("n_batch", None) or 16
        self.n_thread = params.get("n_thread", None) or 4

    def completions(self, args):
        params = llamacpp.LlamaContextParams()
        params.seed = -1

        model = llamacpp.LlamaContext(self.model_path, params)

        # args can be None, so need the "or" part to handle
        top_k = args.get("top_k", None) or 1
        top_p = args.get("top_p", None) or 1
        temp = args.get("temperature", None) or 1.0
        echo = args.get("echo", None) or False
        max_tokens = args.get("max_tokens", None) or 16
        repeat_penalty = 1.3

        prompt = args["prompt"]
        prompt_tokens = model.str_to_token(prompt, True).tolist()
        n_past = self.eval_token(model, prompt_tokens[:-1], 0)

        result = prompt if echo else ""
        token = prompt_tokens[-1]
        finish_reason = "length"
        for i in range(max_tokens):
            model.eval(array.array("i", [token]), 1, n_past, self.n_thread)

            token = model.sample_top_p_top_k(
                array.array("i", []), top_k, top_p, temp, repeat_penalty
            )
            if token == 2:
                # EOS
                finish_reason = "stop"
                break

            text = model.token_to_str(token)
            result += text
            n_past += 1

        c_prompt_tokens = len(prompt_tokens)
        return {
            "id": get_uuid(),
            "object": "text_completion",
            "created": get_timestamp(),
            "model": args["model"],
            "choices": [
                {
                    "text": result,
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

    def embeddings(self, args):
        params = llamacpp.LlamaContextParams()
        params.seed = -1
        params.embedding = True

        model = llamacpp.LlamaContext(self.model_path, params)

        inputs = args["input"]
        if inputs is str:
            inputs = [inputs]
            is_array = False
        else:
            is_array = True
        embeds = []

        for i in inputs:
            prompt_tokens = model.str_to_token(i, True)
            n_past = self.eval_token(model, prompt_tokens, 0)

            embed = unpack_cfloat_array(model.get_embeddings())
            embeds.append(embed)

        if not is_array:
            embeds = embeds[0]

        c_prompt_tokens = len(prompt_tokens)
        return {
            "object": "list",
            "data": [{"object": "embedding", "embedding": embed, "index": 0}],
            "model": args["model"],
            "usage": {
                "prompt_tokens": c_prompt_tokens,
                "total_tokens": c_prompt_tokens,
            },
        }

    def eval_token(self, model, tokens, n_past):
        l = len(tokens)
        for s in range(0, l, self.n_batch):
            e = min(l, s + self.n_batch)
            p = e - s
            model.eval(array.array("i", tokens[s:e]), p, n_past, self.n_thread)
            n_past += p
        return n_past
