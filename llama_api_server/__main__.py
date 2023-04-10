from os import system
import sys

system(
    "FLASK_CONFIG_YAML=$(pwd)/config.yml FLASK_TOKEN_FILE=$(pwd)/tokens.txt flask -A llama_api_server.app run "
    + " ".join(sys.argv[1:])
)
