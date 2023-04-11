import yaml

_config = None


def load_config(app):
    with open(app.config["CONFIG_YAML"], "r") as fd:
        global _config
        _config = yaml.safe_load(fd)


def get_config():
    if _config is not None:
        return _config
    else:
        raise AttributeError("Config not initialized")
