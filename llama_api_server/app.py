from flask import Flask, request
from llama_api_server.model_pool import get_model

app = Flask(__name__)
app.config.from_prefixed_env()


def completions(name, args):
    model = get_model(app, "completions", name)
    return model.completions(args)


def embeddings(name, args):
    model = get_model(app, "embeddings", name)
    return model.embeddings(args)


@app.route("/v1/openai/deployments/<deployment>/completions", methods=["POST"])
@app.route("/v1/engines/<deployment>/completions", methods=["POST"])
def completions_openai(deployment):
    data = request.json
    data["model"] = deployment
    return completions(deployment, data)


@app.route("/v1/completions", methods=["POST"])
def completions_v1():
    data = request.json
    name = data["model"]
    return completions(name, data)


@app.route("/v1/openai/deployments/<deployment>/embeddings", methods=["POST"])
@app.route("/v1/engines/<deployment>/embeddings", methods=["POST"])
def embeddings_openai(deployment):
    data = request.json
    data["model"] = deployment
    return embeddings(deployment, data)


@app.route("/v1/embeddings", methods=["POST"])
def embeddings_v1():
    data = request.json
    name = data["model"]
    return embeddings(name, data)
