# llama.cpp for Qwen3.8-Flash-Next

A private fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) dedicated to
Qwen3.8-Flash-Next on one machine. It is not meant to be merged upstream. The branch
`qwen3.8-flash-next` runs the DwarfStar-packed
[`antirez/qwen3.8-flash-next-gguf`](https://huggingface.co/antirez/qwen3.8-flash-next-gguf)
Q2 in llama.cpp, with its built-in MTP block, on a single 12 GiB CUDA GPU with the
routed experts in system RAM (`-cmoe`). For everything else, see the
[upstream README](https://github.com/ggml-org/llama.cpp/blob/master/README.md).

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
| cache + padded down | this branch | zero-pads the cached-expert `down` input too; without it `--moe-expert-cache` aborts on the Q2 file |
| load-time checks | this branch | rejects default M-RoPE sections that do not fit `n_rot`, and `down_exps` padding other than the quant block size |
| n-gram prefetch | this branch | `POSIX_FADV_RANDOM` on the lazy reader and `POSIX_FADV_WILLNEED` on each batch of rows, so cold rows cost one disk round trip per token instead of one per row |
| quantize copy check | this branch | `llama-quantize` no longer asks for an imatrix for tensors kept in their current type, so single tensor groups can be requantized |

The five PRs are unmerged upstream and may change or be superseded; each is one
commit here so it can be dropped once it lands. The compatibility patches were
written with an AI assistant and have not been reviewed upstream. `test-gguf` covers
the aligned gap, `test-qwen4exp-hparams` the u64 arrays and the M-RoPE default.

The MoE expert cache is set with `--moe-expert-cache N` (env `LLAMA_ARG_MOE_EXPERT_CACHE`);
the `LLAMA_MOE_CACHE_SLOTS` name in its commit message and log is from an older revision.
It only acts on single-token decode, so with `draft-mtp` it skips the multi-token
verification batches. On the tested configuration below, 8 slots (all the free VRAM
allows) gave 25.8 tok/s greedy decode against 26.7 without the cache, so it stays off.

## Build

```sh
docker build \
  -f .devops/cuda.Dockerfile \
  --target server \
  --build-arg UBUNTU_VERSION=24.04 \
  --build-arg CUDA_VERSION=12.8.1 \
  --build-arg CUDA_DOCKER_ARCH=120 \
  -t llamacpp-cuda:qwen3.8-flash-next \
  .
```

## Tested configuration

i5-13500 VM (6 P-cores with SMT as 12 vCPUs), 91 GiB RAM, RTX 5070 12 GiB (PCIe 4.0 x16), model on NVMe. The server runs with:

```sh
llama-server \
  --model /gguf/Qwen3.8-Flash-Next-Q2.gguf \
  --port 8080 \
  -c 98304 \
  -b 2048 \
  --ubatch-size 2048 \
  -ngl 99 \
  -cmoe \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --no-kv-unified \
  -np 1 \
  --cpu-mask 0 \
  --cpu-range 0-5 \
  --cpu-strict 1 \
  --threads 6 \
  --threads-batch 6 \
  --load-mode none \
  --no-repack \
  --fit off \
  -fa on \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.6 \
  --cache-ram 1024 \
  --jinja \
  --temp 1.0 \
  --top-p 1.0 \
  --min-p 0.01 \
  --metrics
```

Measured with the server's default sampling (temp 1.0): about 23-24 tok/s decode at
short context and 21-23 tok/s at 23K, 855-877 tok/s prefill of a 23K prompt; peak VRAM 11.3 GB.
`--spec-draft-p-min` matters: with the defaults, sampled drafts were accepted only
33-42% and decode fell below the no-speculation rate. At `-c 131072` the MTP context
does not fit in 12 GiB, and neither does `--ubatch-size 3072` at `-c 98304` (a 4096
ubatch would give about +20% prefill). Correctness was checked with coherent answers
and needle retrieval at 10/50/90% of a 32K prompt; output quality of the 2-bit model
was not evaluated.

Other results on this machine:

- 12 threads (the SMT siblings too) decode slower than 6; which 6 vCPUs does not matter.
- Without MTP, short-context decode is the same (22.4 vs 22.5 tok/s); MTP helps at depth.
  `--spec-draft-n-max 3` is slower.

## Where the decode time goes

`llama bench` tg64 without MTP takes about 39.5 ms per token:

| Part | ms/token | Notes |
|--|--:|--|
| routed experts on the CPU (6 threads) | ~17 | scales with threads, not with SMT: IQ2_XXS compute, not memory bandwidth |
| GPU kernels | ~13 | `mul_mat_vec_q` 8.5 ms near peak bandwidth; small ops ~4.5 ms |
| CPU/GPU hand-offs and serial parts | ~10 | two waits per MoE layer, 106 per token |

GPU numbers are from an `nsys` 2026.3 trace (`--cuda-graph-trace=node`), the CPU split
from the 3- and 6-thread decode rates. Fused Qwen kernels like DwarfStar's can only
reach the ~4.5 ms of small GPU ops. The n-gram reads cost ~1.9 ms per token before
the prefetch commit, with cold rows at ~120 us each.

## Optional: Q8_0 hyper-connection mixers

The 196 `hc_{attn,ffn}_{down,up}` tensors are F16 (1.2 GiB, read every token). This
converts only those to Q8_0 and copies every other tensor bit for bit (needs the
quantize copy check commit):

```sh
llama quantize \
  --tensor-type "token_embd=bf16" \
  --tensor-type "blk\.48\.ffn_(gate|up)_exps=q4_k" --tensor-type "blk\.48\.ffn_down_exps=mxfp4" \
  --tensor-type "ffn_(gate|up)_exps=iq2_xxs" --tensor-type "ffn_down_exps=q2_k" \
  --tensor-type "hc_(attn|ffn)_(down|up)=q8_0" \
  --tensor-type "(hc_(attn|ffn)_inject|output_hc_(down|up)|hc_head_(down|up))=f16" \
  --tensor-type "(ffn_gate_inp|ssm_alpha|ssm_beta|ssm_conv1d|ple_conv1d)=f32" \
  Qwen3.8-Flash-Next-Q2.gguf Qwen3.8-Flash-Next-Q2-hcq8.gguf q8_0 6
```

It saves 574 MiB of VRAM and gives about +3% short and +5-8% long-context decode. The
first 128 greedy tokens match the F16 file; no other quality check has been run, so it
is not the tested configuration.
