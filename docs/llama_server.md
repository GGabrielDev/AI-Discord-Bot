# llama-server Configuration Guide

Recommended configurations for running the Autonomous Research Agent with `llama-server` and the `unsloth/gemma-4-E4B-it-GGUF:UD-Q8_K_XL` model.

## Quick Start (Max Context: 128K)

Recommended command for running **Gemma 4** or **DeepSeek-R1** at maximum research capacity.

```bash
llama-server \
  -m /path/to/model-Q8_K_XL.gguf \
  --jinja \
  -ngl 99 \
  -c 131072 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --host 0.0.0.0 \
  --port 8080
```
> [!NOTE]
> For **DeepSeek-R1 (Qwen)**, you can omit the `--jinja` flag as it uses standard ChatML templates.

## Flag Breakdown

| `-m` | Path to `.gguf` | The quantized model file |
| `--jinja` | (flag) | **Required for Gemma 4**. Omit for DeepSeek/Qwen. |
| `-ngl 99` | 99 | Offload all layers to GPU. |
| `-c 131072` | 128K | **Max Context**. Matches the bot's default `LLM_CONTEXT_WINDOW`. |
| `--flash-attn on` | `on` | Enables Flash Attention. **Required** for 128K stability. |
| `--cache-type-k q8_0` | q8_0 | 8-bit KV Cache. **Mandatory** for 128K to fit in VRAM. |
| `--cache-type-v q8_0` | q8_0 | 8-bit V-cache quantization. |
| `--host 0.0.0.0` | Bind all interfaces | Makes the server accessible to the bot. |
| `--port 8080` | 8080 | Must match `LLM_API_BASE` in your `.env`. |

## Context Window Sizing

The research agent sends varying amounts of context depending on the task:

| Task | Typical Context | Recommended `-c` |
|------|----------------|-------------------|
| Deep Page Crawling | 10K-50K tokens | 128K |
| Massive RAG Synthesis | 20K-80K tokens | 128K |
| Multihop Gap Seeking | 5K-30K tokens | 128K |

**Recommendation:** Use `-c 131072` (128K). This allows the research agent to ingest massive amounts of scraped data into a single synthesis pass, drastically improving the quality of the final report.

**VRAM vs Context counts (8B Models with q8_0 KV Cache):**
- **32K Context**: ~2GB VRAM used for KV cache.
- **64K Context**: ~4GB VRAM used for KV cache.
- **128K Context**: ~8GB VRAM used for KV cache.

> [!TIP]
> If using an 8B model (like DeepSeek-R1-8B) on a 12GB or 16GB VRAM card, **128K context is easily achievable** using the `q8_0` flags.

## KV Cache Quantization Options

### Standard llama.cpp (upstream)

Available cache types: `f32`, `f16` (default), `bf16`, `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`

| Config | Quality | VRAM Savings | Notes |
|--------|---------|-------------|-------|
| `f16` / `f16` | Best | None (baseline) | Default, no quality loss |
| `q8_0` / `q8_0` | Near-lossless | ~50% KV cache | **Recommended starting point** |
| `q8_0` / `q4_0` | Good | ~62% KV cache | V-cache is less sensitive to quantization |
| `q4_0` / `q4_0` | Acceptable | ~75% KV cache | May degrade complex reasoning |

### RotorQuant (experimental, maximum compression)

> [!WARNING]  
> **Gemma 4 Compatibility Note:** Because `turboquant` is an experimental research fork, it is often lagging behind the main `llama.cpp` repository. As of right now, the `feature/planarquant-kv-cache` branch does not yet support the `gemma4` model architecture (it throws `unknown model architecture: 'gemma4'`). If you are running Gemma 4, you must stick with the vanilla `llama-server` and use the standard `q8_0` KV cache options until the developer merges the upstream updates.

Because `turboquant` is just a fork of the regular `llama-server`, **it uses the exact same flags** (like `-c`, `--flash-attn`, and `--jinja`). The only difference is that turboquant mathematically unlocks the `iso3` options for your `--cache-type` flags instead of being limited to `q8_0`.

```bash
# 1. Clone the RotorQuant fork
git clone https://github.com/johndpope/llama-cpp-turboquant.git
cd llama-cpp-turboquant && git checkout feature/planarquant-kv-cache

# 2. Build the Server
# 👉 FOR AMD GPUs / APUs (ROCm on Linux):
cmake -B build -DGGML_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# 👉 FOR NVIDIA (Windows/Linux):
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# 👉 FOR MACS (Apple Silicon):
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# 👉 UNIVERSAL FALLBACK (Vulkan for unsupported GPUs):
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# 3. Run with RotorQuant KV cache (10x compression!)
./build/bin/llama-server \
  -m /path/to/gemma-4-E4B-it-UD-Q8_K_XL.gguf \
  --jinja \
  -ngl 99 \
  -c 32768 \
  --flash-attn on \
  --cache-type-k iso3 \
  --cache-type-v iso3 \
  --host 0.0.0.0 \
  --port 8080
```

| Config | PPL (Llama 3.1 8B) | Compression | Speed |
|--------|------------------:|:-----------:|:-----:|
| `f16` / `f16` | ~6.5 | 1x | Baseline |
| `q8_0` / `q8_0` | ~6.6 | 2x | ~Same |
| `iso3` / `iso3` | 6.91 | 10.3x | 28% faster decode, 5.3x faster prefill |
| `planar3` / `f16` | ~6.5 | 5.1x K-only | Zero PPL loss on K-cache |

**When to use RotorQuant:**
- You need 65K+ context on limited VRAM
- You want faster decode speed
- You're comfortable maintaining a custom llama.cpp build

**When to stick with upstream:**
- Simplicity is more important
- 32K context is sufficient (it usually is)
- You want easy updates via official llama.cpp releases

## Performance Tips

1. **GPU Offload:** Always use `-ngl 99` if the model fits in VRAM. Partial GPU offload (e.g., `-ngl 20`) is much slower than full offload.

2. **Batch Size:** The default batch size is usually fine. If you're doing many sequential summarizations, the model processes them one at a time anyway.

3. **Parallel Requests:** The research agent makes sequential LLM calls (plan → scrape → summarize → evaluate). There's no benefit to enabling parallel slots unless you're also running the legacy `!prompt` command simultaneously.

4. **Temperature 0 for debugging:** If the model produces erratic output, test with `temperature=0` in your `.env` or code to rule out sampling randomness.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Connection refused" | Ensure llama-server is running and `LLM_API_BASE` matches the `--host:--port` |
| Model hangs on generation | Increase `LLM_TIMEOUT` in `.env`. Some summarizations take 60s+ |
| JSON parsing failures | The client retries at `temperature=0.0`. If persistent, the model may need a different quantization |
| Out of VRAM | Reduce `-c` (context), use `q8_0` KV cache, or reduce `-ngl` |
| Garbled output | Ensure `--jinja` is set — Gemma 4 requires it for proper chat template |
