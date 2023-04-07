from flask import Flask, request
from llama_api_server.model_pool import get_model

app = Flask(__name__)
app.config.from_prefixed_env()


def completions(name, args):
    model = get_model(app, "completions", name)
    return model.completions(args)


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
