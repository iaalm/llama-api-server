🎭🦙 llama-api-server
=======

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Release](https://github.com/iaalm/llama-api-server/actions/workflows/release.yml/badge.svg)](https://github.com/iaalm/llama-api-server/actions/workflows/release.yml)
[![PyPI version](https://badge.fury.io/py/llama-api-server.svg)](https://badge.fury.io/py/llama-api-server)

This project is under active deployment. Breaking changes could be made any time.

Llama as a Service! This project try to build a REST-ful API server compatible to OpenAI API using open source backends like llama/llama2.

With this project, many common GPT tools/framework can compatible with your own model.

# 🚀Get started

## Prepare model

### llama.cpp
If you you don't have quantized llama.cpp, you need to follow [instruction](https://github.com/ggerganov/llama.cpp#usage) to prepare model.

### pyllama
If you you don't have quantize pyllama, you need to follow [instruction](https://github.com/juncongmoo/pyllama#-quantize-llama-to-run-in-a-4gb-gpu) to prepare model.


## Install
Use following script to download package from [PyPI](https://pypi.org/project/llama-api-server) and generates model config file `config.yml` and security token file `tokens.txt`.
```
pip install llama-api-server

# to run wth pyllama
pip install llama-api-server[pyllama]

cat > config.yml << EOF
models:
  completions:
    # completions and chat_completions use same model
    text-ada-002:
      type: llama_cpp
      params:
        path: /absolute/path/to/your/7B/ggml-model-q4_0.bin
    text-davinci-002:
      type: pyllama_quant
      params:
        path: /absolute/path/to/your/pyllama-7B4b.pt
    text-davinci-003:
      type: pyllama
      params:
        ckpt_dir: /absolute/path/to/your/7B/
        tokenizer_path: /absolute/path/to/your/tokenizer.model
      # keep to 1 instance to speed up loading of model
  embeddings:
    text-embedding-davinci-002:
      type: pyllama_quant
      params:
        path: /absolute/path/to/your/pyllama-7B4b.pt
      min_instance: 1
      max_instance: 1
      idle_timeout: 3600
    text-embedding-ada-002:
      type: llama_cpp
      params:
        path: /absolute/path/to/your/7B/ggml-model-q4_0.bin
EOF

echo "SOME_TOKEN" > tokens.txt

# start web server
python -m llama_api_server
# or visible across the network
python -m llama_api_server --host=0.0.0.0

```

## Call with openai-python
```
export OPENAI_API_KEY=SOME_TOKEN
export OPENAI_API_BASE=http://127.0.0.1:5000/v1

openai api completions.create -e text-ada-002 -p "hello?"
# or using chat
openai api chat_completions.create -e text-ada-002 -g user "hello?"
# or calling embedding
curl -X POST http://127.0.0.1:5000/v1/embeddings -H 'Content-Type: application/json' -d '{"model":"text-embedding-ada-002", "input":"It is good."}'  -H "Authorization: Bearer SOME_TOKEN"
```

# 🛣️Roadmap

### Tested with
- [X] [openai-python](https://github.com/openai/openai-python)
    - [X] OPENAI\_API\_TYPE=default
    - [X] OPENAI\_API\_TYPE=azure
- [X] [llama-index](https://github.com/jerryjliu/llama_index)

### Supported APIs
- [X] Completions
    - [X] set `temperature`, `top_p`, and `top_k`
    - [X] set `max_tokens`
    - [X] set `echo`
    - [ ] set `stop`
    - [ ] set `stream`
    - [ ] set `n`
    - [ ] set `presence_penalty` and `frequency_penalty`
    - [ ] set `logit_bias`
- [X] Embeddings
    - [X] batch process
- [X] Chat
    - [ ] Prefix cache for chat
- [ ] List model

### Supported backends
- [X] [llama.cpp](https://github.com/ggerganov/llama.cpp) via [llamacpp-python](https://github.com/thomasantony/llamacpp-python)
- [X] [llama](https://github.com/facebookresearch/llama) via [pyllama](https://github.com/juncongmoo/pyllama)
    - [X] Without Quantization
    - [X] With Quantization
    - [X] Support LLAMA2

### Others
- [X] Performance parameters like `n_batch` and `n_thread`
- [X] Token auth
- [ ] Documents
- [ ] Intergration tests
- [ ] A tool to download/prepare pretrain model
- [ ] Make config.ini and token file configable
