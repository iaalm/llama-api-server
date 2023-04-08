import yaml
from llama_api_server.models.llama_cpp import LlamaCpp
from collections import defaultdict

_pool = defaultdict(lambda: defaultdict(list))

MODEL_TYPE_MAPPING = {"llama_cpp": LlamaCpp}


def load_config(app):
    with open(app.config["CONFIG_YAML"], "r") as fd:
        return yaml.safe_load(fd)


def get_model(app, kind, name):
    models = _pool[kind][name]
    config = load_config(app)["models"][kind][name]
    model = MODEL_TYPE_MAPPING[config["type"]](config["params"])
    return model
