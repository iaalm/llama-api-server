import llamacpp
from llama_api_server.utils import get_uuid, get_timestamp


class LlamaCpp:
    def __init__(self, params):
        self.model_path = params["path"]

    def completions(self, args):
        print(args)
        params = llamacpp.InferenceParams()
        params.seed = -1
        params.n_threads = 4

        params.repeat_last_n = 64
        params.n_batch = 8
        params.top_k = 40
        params.top_p = 0.95
        params.temp = 0.7
        params.repeat_penalty = 1.3
        params.use_mlock = False
        params.memory_f16 = False
        params.n_ctx = 512

        params.path_model = self.model_path
        model = llamacpp.LlamaInference(params)

        prompt = args["prompt"]
        prompt_tokens = model.tokenize(prompt, True)
        model.update_input(prompt_tokens)

        model.ingest_all_pending_input()

        result = ""
        for i in range(20):
            model.eval()
            token = model.sample()
            text = model.token_to_str(token)
            result += text
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
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        }
