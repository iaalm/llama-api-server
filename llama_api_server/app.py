from functools import wraps, cache
from flask import Flask, request
from llama_api_server.model_pool import get_model

app = Flask(__name__)
app.config.from_prefixed_env()


@cache
def getValidTokens():
    token_file = app.config["TOKEN_FILE"]
    with open(token_file, "r") as fd:
        return [i.strip() for i in fd]


def requireToken(f):
    @wraps(f)
    def inner(*args, **kwargs):
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
    model = get_model(app, "completions", name)
    return model.completions(args)


def embeddings(name, args):
    model = get_model(app, "embeddings", name)
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
