from os import system

system("FLASK_CONFIG_YAML=$(pwd)/config.yml flask -A llama_api_server.app run")
