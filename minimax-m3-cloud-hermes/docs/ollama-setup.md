# Local LLM with Ollama

The agent supports any OpenAI-shaped `/v1/chat/completions` endpoint as a
swap-in for the stub LLM. The most common one for local dev is
[Ollama](https://ollama.com), which serves qwen, llama, mistral, etc. on
`http://127.0.0.1:11434/v1`.

## Install ollama (Linux ARM64)

The prebuilt `ollama-linux-arm64.tgz` is a stub that requires the
`llama-server` runtime to be co-located. On this sandbox we got it
working like this:

```bash
# 1. Install zstd (needed to decompress the release archive)
apt-get install -y zstd

# 2. Download the release tarball from the GitHub releases page
mkdir -p /root/ollama-install && cd /root/ollama-install
curl -L -o ollama.tar.zst \
  https://github.com/ollama/ollama/releases/download/v0.30.3/ollama-linux-arm64.tar.zst

# 3. Extract and install
zstd -d ollama.tar.zst -o ollama.tar
tar -xf ollama.tar
cp bin/ollama /usr/local/bin/ollama
chmod +x /usr/local/bin/ollama
mkdir -p /usr/local/lib/ollama
cp lib/ollama/llama-server /usr/local/lib/ollama/llama-server
chmod +x /usr/local/lib/ollama/llama-server
```

## Start the daemon

The `llama-server` binary dlopens `libggml-cpu-*.so` from the current
working directory (not from `LD_LIBRARY_PATH`), so the daemon must be
started with `cwd = /root/ollama-install/lib/ollama`. The repo ships
`scripts/ollama_serve.sh`:

```bash
scripts/ollama_serve.sh
# or in the background:
nohup scripts/ollama_serve.sh > /tmp/ollama.log 2>&1 &
```

The daemon listens on `127.0.0.1:11434` (set `OLLAMA_HOST` to change).

## Pull a model

```bash
ollama pull qwen2.5:0.5b      # 379 MB; default, fast, less coherent
ollama pull qwen2.5:1.5b      # 940 MB; opt-in via env override
```

## Point the agent at ollama

```bash
export SOCCER_AGENT_LLM_PROVIDER=ollama
# optional overrides:
export SOCCER_AGENT_LLM_BASE_URL=http://127.0.0.1:11434/v1
export SOCCER_AGENT_LLM_MODEL=qwen2.5:0.5b
# (no API key needed for local ollama)
```

Then run the agent as usual:

```bash
soccer-agent predict --team-home "Real Madrid" --team-away "Man City"
```

The agent will route LLM calls to your local daemon.

## Performance notes

On the sandbox (ARM64, no GPU, 0.5B model):

- prompt eval: ~1.0 tok/s
- generation: ~0.05 tok/s
- a 50-token generation takes ~15 minutes

For prompt iteration, prefer the **stub LLM** (instant, deterministic)
and only flip to ollama when you want to test the full E2E pipeline
with a real model. On a dev machine with M-series silicon or a GPU,
expect 5-50x speedups.

## Live test

`tests/test_ollama_live.py` contains 4 tests that hit the live daemon.
They are auto-skipped if the daemon is unreachable:

```bash
pytest tests/test_ollama_live.py -m ollama -v
# (takes ~2-3 min on sandbox; <30s on a dev machine)
```

The default `pytest` run excludes them (`-m "not ollama"` is implicit)
so the suite stays green on machines without ollama.
