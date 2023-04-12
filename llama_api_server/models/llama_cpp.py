import llamacpp
import array
from llama_api_server.utils import get_uuid, get_timestamp, unpack_cfloat_array


def _create_llama_model(model_path, embedding=False):
    params = llamacpp.LlamaContextParams()
    params.seed = -1
    params.embedding = embedding
    model = llamacpp.LlamaContext(model_path, params)
    return model


def _eval_token(model, tokens, n_past, n_batch, n_thread):
    l = len(tokens)
    for s in range(0, l, n_batch):
        e = min(l, s + n_batch)
        p = e - s
        model.eval(array.array("i", tokens[s:e]), p, n_past, n_thread)
        n_past += p
    return n_past


class LlamaCppCompletion:
    def __init__(self, params):
        model_path = params["path"]
        self.model = _create_llama_model(model_path, False)
        self.n_batch = params.get("n_batch", None) or 2
        self.n_thread = params.get("n_thread", None) or 4

    def completions(self, args):
        # args can be None, so need the "or" part to handle
        top_k = args["top_k"]
        top_p = args["top_p"]
        temp = args["temperature"]
        echo = args["echo"]
        max_tokens = args["max_tokens"]
        suffix = args["suffix"]
        repeat_penalty = 1.3

        prompt = args["prompt"]
        prompt_tokens = self.model.str_to_token(prompt, True).tolist()
        n_past = _eval_token(
            self.model, prompt_tokens[:-1], 0, self.n_batch, self.n_thread
        )

        result = prompt if echo else ""
        token = prompt_tokens[-1]
        finish_reason = "length"
        for i in range(max_tokens):
            self.model.eval(array.array("i", [token]), 1, n_past, self.n_thread)

            token = self.model.sample_top_p_top_k(
                array.array("i", []), top_k, top_p, temp, repeat_penalty
            )
            if token == 2:
                # EOS
                finish_reason = "stop"
                break

            text = self.model.token_to_str(token)
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


class LlamaCppEmbedding:
    def __init__(self, params):
        model_path = params["path"]
        self.model = _create_llama_model(model_path, True)
        self.n_batch = params.get("n_batch", None) or 2
        self.n_thread = params.get("n_thread", None) or 4

    def embeddings(self, args):
        inputs = args["input"]
        if inputs is str:
            inputs = [inputs]
            is_array = False
        else:
            is_array = True
        embeds = []

        for i in inputs:
            prompt_tokens = self.model.str_to_token(i, True)
            n_past = _eval_token(
                self.model, prompt_tokens, 0, self.n_batch, self.n_thread
            )

            embed = unpack_cfloat_array(self.model.get_embeddings())
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
