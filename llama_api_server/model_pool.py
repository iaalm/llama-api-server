import datetime
from collections import defaultdict
from functools import cache
from threading import Lock
from llama_api_server.models.llama_cpp import LlamaCppCompletion, LlamaCppEmbedding
from llama_api_server.models.pyllama import PyLlamaCompletion
from llama_api_server.models.pyllama_quant import PyLlamaQuantCompletion
from .config import get_config

# Eventhrough python is not good at multi-threading, but must work is done by backend,
# support multi thread with flask threaded mode may be a good idea.


_pool = defaultdict(lambda: defaultdict(list))
_pool_count = defaultdict(lambda: defaultdict(int))
_lock = Lock()

MODEL_TYPE_MAPPING = {
    "embeddings": {"llama_cpp": LlamaCppEmbedding},
    "completions": {
        "llama_cpp": LlamaCppCompletion,
        "pyllama": PyLlamaCompletion,
        "pyllama_quant": PyLlamaQuantCompletion,
    },
}


class _ModelInPool:
    def __init__(self, model, kind, name):
        self._model = model
        self.name = name
        self.kind = kind
        self.last_used = None

    def __enter__(self):
        return self._model

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.last_used = datetime.datetime.now()
        _return_model(self, self.kind, self.name)


def get_model(kind, name):
    with _lock:
        p = _pool[kind][name]
        if len(p) > 0:
            model = p.pop()
        else:
            config = get_config()["models"][kind][name]
            if _pool_count[kind][name] < config.get("max_instances", 1):
                inner_model = MODEL_TYPE_MAPPING[kind][config["type"]](config["params"])
                model = _ModelInPool(inner_model, kind, name)
                _pool_count[kind][name] += 1
            else:
                # deep learning models run relatively long time, it might be
                # better to tell client instead of waiting for lock
                # TODO: better error handling with customized error and 409
                raise TimeoutError()
        return model


def _return_model(model, kind, name):
    with _lock:
        if get_config()["models"][kind][name].get("idle_timeout", 0) != 0:
            _pool[kind][name].append(model)
        else:
            _pool_count[kind][name] -= 1


def _retention_model():
    all_config = get_config()
    with _lock:
        for kind in _pool:
            for name in _pool[kind]:
                _return_model_raw(kind, name)


def _return_model_raw(kind, name):
    config = get_config()["models"][kind][name]
    p = _pool[kind][name]
    oldest_time = datetime.datetime.now() - datetime.timedelta(
        seconds=config.get("idle_timeout", 0)
    )
    min_instance = config.get("min_instances", 0)

    # delete redudent models
    while _pool_count[kind][name] > min_instance:
        if p[0].last_used < oldest_time:
            p.pop(0)
            _pool_count[kind][name] -= 1
        else:
            break
