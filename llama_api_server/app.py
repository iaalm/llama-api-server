from functools import wraps, cache
from flask import Flask, request
from llama_api_server.model_pool import get_model
from .config import load_config, get_config
from .chat_engine import dialog_to_llama_prompt

app = Flask(__name__)
app.config.from_prefixed_env()
load_config(app)


@cache
def isAuthEnabled():
    token_file = app.config.get("TOKEN_FILE", None)
    return token_file is not None

@cache
def getValidTokens():
    token_file = app.config["TOKEN_FILE"]
    with open(token_file, "r") as fd:
        return [i.strip() for i in fd]


def requireToken(f):
    @wraps(f)
    def inner(*args, **kwargs):
        if not isAuthEnabled():
            return f(*args, **kwargs)
        token = request.headers.get("Authorization")
        if token is not None:
            # OpenAI style token
            token_type = token.split(" ")[0]
            token_value = token[len(token_type) + 1 :]
        else:
            # Azure style key
            token_value = request.headers.get("Api-Key")
            token_type = "Bearer"
        if token_type == "Bearer" and token_value in getValidTokens():
            return f(*args, **kwargs)
        else:
            return "Invalid Authentication", 401

    return inner


def completions(name, args):
    args["top_k"] = args.get("top_k", None) or 1
    args["top_p"] = args.get("top_p", None) or 1.0
    args["temperature"] = args.get("temperature", None) or 1.0
    args["echo"] = args.get("echo", None) or False
    args["max_tokens"] = args.get("max_tokens", None) or 16
    args["suffix"] = args.get("suffix", None) or ""
    with get_model("completions", name) as model:
        return model.completions(args)


def chat_completions(name, args):
    args["top_k"] = args.get("top_k", None) or 1
    args["top_p"] = args.get("top_p", None) or 1.0
    args["temperature"] = args.get("temperature", None) or 1.0
    args["max_tokens"] = args.get("max_tokens", None) or 16
    args["echo"] = False
    args["suffix"] = ""
    prompt = dialog_to_llama_prompt(args["messages"])
    args["prompt"] = prompt
    with get_model("completions", name) as model:
        res = model.completions(args)

    res["choices"][0]["message"] = {
        "role": "assistant",
        "content": res["choices"][0]["text"],
    }
    res["choices"][0]["text"] = None
    res["choices"][0]["logprobs"] = None
    return res


def embeddings(name, args):
    with get_model("embeddings", name) as model:
        return model.embeddings(args)


@app.route("/v1/openai/deployments/<deployment>/completions", methods=["POST"])
@app.route("/v1/engines/<deployment>/completions", methods=["POST"])
@requireToken
def completions_openai(deployment):
    data = request.json
    data["model"] = deployment
    return completions(deployment, data)


@app.route("/v1/completions", methods=["POST"])
@requireToken
def completions_v1():
    data = request.json
    name = data["model"]
    return completions(name, data)


@app.route("/v1/openai/deployments/<deployment>/embeddings", methods=["POST"])
@app.route("/v1/engines/<deployment>/embeddings", methods=["POST"])
@requireToken
def embeddings_openai(deployment):
    data = request.json
    data["model"] = deployment
    return embeddings(deployment, data)


@app.route("/v1/embeddings", methods=["POST"])
@requireToken
def embeddings_v1():
    data = request.json
    name = data["model"]
    return embeddings(name, data)


@app.route("/v1/openai/deployments/<deployment>/chat/completions", methods=["POST"])
@app.route("/v1/engines/<deployment>/chat/completions", methods=["POST"])
@requireToken
def chat_completions_openai(deployment):
    data = request.json
    data["model"] = deployment
    return chat_completions(deployment, data)


@app.route("/v1/chat/completions", methods=["POST"])
@requireToken
def chat_completions_v1():
    data = request.json
    name = data["model"]
    return chat_completions(name, data)
