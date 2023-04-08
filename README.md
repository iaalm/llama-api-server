llama-api-server
=======

This project is under active deployment. Breaking changes could be made any time.

Llama as a Service! This project try to build a REST-ful API server compatible to OpenAI API using open source backends like llama.

#### Tested with
- [X] openai-python
    - [X] OPENAI\_API\_TYPE=default
    - [X] OPENAI\_API\_TYPE=azure

#### Supported APIs
- [X] Completions
    - [X] set `temperature`, `top\_p`, and `top\_k`
    - [X] set `max\_tokens`
    - [ ] set `stop`
    - [ ] set `stream`
    - [ ] set `n`
    - [ ] set `presence\_penalty` and `frequency\_penalty`
    - [ ] set `logit\_bias`
- [X] Embeddings
    - [X] batch process
- [ ] Chat

#### Supported backed
- [X] [llama.cpp](https://github.com/ggerganov/llama.cpp) via [llamacpp-python](https://github.com/thomasantony/llamacpp-python)

#### Others
- [ ] Documents
- [ ] Token auth
- [ ] Intergration tests
- [ ] Performance parameters like `n_batch` and `n_thread`
- [ ] A tool to download/prepare pretrain model

## Get started

### Install
```
pip install llama-api-server
echo > config.yml << EOF
models:
  completions:
    text-davinci-003:
      type: llama_cpp
      params:
        path: /absolute/path/to/your/7B/ggml-model-q4_0.bin
  embeddings:
    text-embedding-ada-002:
      type: llama_cpp
      params:
        path: /absolute/path/to/your/7B/ggml-model-q4_0.bin
EOF
python -m python -m llama_api_server
```

### Call with openai-python
```
export OPENAI_API_BASE=http://127.0.0.1:5000/v1
openai api completions.create -e text-davinci-003 -p "hello?"
```
