🎭🦙 llama-api-server
=======

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Release](https://github.com/iaalm/llama-api-server/actions/workflows/release.yml/badge.svg)](https://github.com/iaalm/llama-api-server/actions/workflows/release.yml)

This project is under active deployment. Breaking changes could be made any time.

Llama as a Service! This project try to build a REST-ful API server compatible to OpenAI API using open source backends like llama.

# 🚀Get started

## Prepare model

### llama.cpp
If you you don't have quantized llama.cpp, you need to follow [instruction](https://github.com/ggerganov/llama.cpp#usage) to prepare model.

### pyllama
If you you don't have quantize pyllama, you need to follow [instruction](https://github.com/juncongmoo/pyllama#-quantize-llama-to-run-in-a-4gb-gpu) to prepare model.


## Install
```
pip install llama-api-server

# to run wth pyllama
pip install llama-api-server[pyllama]

echo > config.yml << EOF
models:
  completions:
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
      min_instance: 1
      max_instance: 1
  embeddings:
    text-embedding-ada-002:
      type: llama_cpp
      params:
        path: /absolute/path/to/your/7B/ggml-model-q4_0.bin
EOF

echo "SOME_TOKEN" > tokens.txt

# start web server
python -m llama_api_server
```

## Call with openai-python
```
export OPENAI_API_KEY=SOME_TOKEN
export OPENAI_API_BASE=http://127.0.0.1:5000/v1

openai api completions.create -e text-ada-002 -p "hello?"

curl -X POST http://127.0.0.1:5000/v1/embeddings -H 'Content-Type: application/json' -d '{"model":"text-embedding-ada-002", "input":"It is good."}'  -H "Authorization: Bearer SOME_TOKEN"
```

# 🛣️Roadmap

### Tested with
- [X] openai-python
    - [X] OPENAI\_API\_TYPE=default
    - [X] OPENAI\_API\_TYPE=azure

### Supported APIs
- [X] Completions
    - [X] set `temperature`, `top_p`, and `top_k`
    - [X] set `max_tokens`
    - [ ] set `stop`
    - [ ] set `stream`
    - [ ] set `n`
    - [ ] set `presence_penalty` and `frequency_penalty`
    - [ ] set `logit_bias`
- [X] Embeddings
    - [X] batch process
- [ ] Chat
- [ ] List model

### Supported backends
- [X] [llama.cpp](https://github.com/ggerganov/llama.cpp) via [llamacpp-python](https://github.com/thomasantony/llamacpp-python)
- [X] [llama](https://github.com/facebookresearch/llama) via [pyllama](https://github.com/juncongmoo/pyllama)
    - [X] Without Quantization
    - [X] With Quantization

### Others
- [X] Performance parameters like `n_batch` and `n_thread`
- [X] Token auth
- [ ] Documents
- [ ] Intergration tests
- [ ] A tool to download/prepare pretrain model
- [ ] Make config.ini and token file configable
