#!/bin/bash
# Start ollama daemon in the background with the right LD_LIBRARY_PATH and cwd.
# The lib path is needed for the llama-server backend .so files.
# The cwd is needed because libllama.so does dlopen("libggml-cpu-*.so")
# relative to the cwd, not via LD_LIBRARY_PATH.
set -e
mkdir -p /root/.ollama/models
cd /root/ollama-install/lib/ollama
exec env \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/root/.ollama/models \
    LD_LIBRARY_PATH=/root/ollama-install/lib/ollama \
    /usr/local/bin/ollama serve
