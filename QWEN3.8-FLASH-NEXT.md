# llama.cpp for Qwen3.8-Flash-Next

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) dedicated to
Qwen3.8-Flash-Next. The branch `qwen3.8-flash-next` runs the
DwarfStar-packed [`antirez/qwen3.8-flash-next-gguf`](https://huggingface.co/antirez/qwen3.8-flash-next-gguf)
Q2 in llama.cpp, with its built-in MTP block, on a single 12 GiB CUDA GPU with the
routed experts in system RAM (`-cmoe`).

It is llama.cpp **v0.5.0** (`7fe450e1`) plus, in this order:

| Commit | Source | What it does |
|--|--|--|
| NextN/MTP draft head | open PR [#28243](https://github.com/ggml-org/llama.cpp/pull/28243) @ `6fcaa16f` (builds on #27836), squashed | loads the MTP block from the main GGUF; `--spec-type draft-mtp` |
| padded `down_exps` | this branch | accepts `ffn_down_exps` stored `[768, n_embd, n_expert]` for a logical 640 (Q2_K needs 256-blocks) and zero-pads the activation |
| aligned GGUF gap | this branch | accepts the 4096-aligned gap DwarfStar leaves before the n-gram table |
| M-RoPE default | this branch | defaults `qwen4exp.rope.dimension_sections` to `[11, 11, 10, 0]` when absent |
| u64 int arrays | this branch | narrows `compress_ratios` / `ple.layers` stored as u64 to int32 |
| radix TOP_K | open PR [#28671](https://github.com/ggml-org/llama.cpp/pull/28671) @ `ff2b4367` | radix-select top-k for wide rows when CCCL < 3.2 (CUDA 12.x) |
| direct lazy reads | open PR [#29030](https://github.com/ggml-org/llama.cpp/pull/29030) @ `e32c6243` | reads n-gram rows with `pread` instead of mmap page faults |
| MoE expert cache | open PR [#27861](https://github.com/ggml-org/llama.cpp/pull/27861) @ `bccbacdb`, rebased | opt-in `--moe-expert-cache N` (off by default) |
| QSA pooled-key cache | open PR [#28699](https://github.com/ggml-org/llama.cpp/pull/28699) @ `141f3f56` | incremental indexer key cache (`LLAMA_QSA_NO_POOLED_CACHE=1` disables) |

The five PRs are unmerged upstream and may change or be superseded; each is one
commit here so it can be dropped once it lands. The four compatibility patches were
written with an AI assistant and have not been reviewed upstream.

## Build

```sh
docker build -f .devops/cuda.Dockerfile \
  --build-arg UBUNTU_VERSION=24.04 --build-arg CUDA_VERSION=12.8.1 --build-arg CUDA_DOCKER_ARCH=120 \
  -t llamacpp-cuda:qwen3.8-flash-next .
```

## Tested configuration

i5-13500 (6 P-cores), 91 GiB RAM, RTX 5070 12 GiB, model on NVMe:

```
--model Qwen3.8-Flash-Next-Q2.gguf -c 98304 -b 2048 --ubatch-size 2048 -ngl 99 -cmoe
--cache-type-k q4_0 --cache-type-v q4_0 -np 1 --no-kv-unified
--threads 6 --cpu-range 0-5 --cpu-strict 1 --load-mode none -fa on
--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.6
```

Measured with the server's default sampling (temp 1.0): about 22.5 tok/s decode at
short context and 20.4 tok/s at 32K, 676-686 tok/s prefill; peak VRAM 11.3 GB.
`--spec-draft-p-min` matters: with the defaults, sampled drafts were accepted only
33-42% and decode fell below the no-speculation rate. At `-c 131072` the MTP context
does not fit in 12 GiB. Correctness was checked with coherent answers and needle
retrieval at 10/50/90% of a 32K prompt; output quality of the 2-bit model was not
evaluated.
