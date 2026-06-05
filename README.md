<div align="center">

# SplitZip: Ultra Fast Lossless KV Compression for Disaggregated LLM Serving

<p>
  <a href="https://arxiv.org/abs/2605.01708">
    <img src="https://img.shields.io/badge/arXiv-2605.01708-b31b1b.svg?logo=arxiv" alt="arXiv">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
</p>

<p>
Yipin Guo and Siddharth Joshi
</p>

</div>

SplitZip is a GPU-friendly lossless compressor for KV cache transfer in
prefill-decode disaggregated LLM serving. It preserves BF16 KV tensors bitwise
while reducing transfer volume and keeping both compression and decompression
on the latency-critical GPU path.

The key observation is that BF16 KV activations have highly redundant exponent
values. SplitZip encodes the most frequent exponent values with fixed 4-bit
codes, keeps sign and mantissa bits exact, and routes rare exponent values
through a sparse escape stream. The released artifact includes the public
Triton codec and reproduction scripts for the BF16 exponent analysis and codec
throughput measurements used in the paper.

## Highlights

- Lossless BF16 KV cache compression with bitwise round-trip recovery.
- Chunk-local Top-16 exponent codebooks with sparse escapes for rare exponents.
- GPU encode and decode kernels implemented in Triton.
- Reproduction scripts for WikiText-2/Qwen3-32B exponent statistics and codec
  throughput.
- Paper-reported codec path performance on real BF16 KV activations:
  613.3 GB/s compression and 2181.8 GB/s decompression.

## Repository Contents

- `codec_gpu.py`: public ChunkLocalSplitZipGPU Triton codec.
- `bench_codec_throughput.py`: encode/decode throughput benchmark for the public
  codec API.
- `analyze_bf16_exponent_entropy_qwen32.py`: BF16 KV exponent entropy and
  Top-16 coverage analysis.
- `splitzip.pdf`: included paper PDF.
- `requirements.txt`: minimal Python runtime dependencies.

## Installation

SplitZip requires a CUDA-capable GPU and a PyTorch/Triton stack compatible with
that GPU. The released scripts were checked against the `yipin_quant` conda
environment, whose direct runtime dependencies are captured in
`requirements.txt`.

```bash
conda create -n splitzip python=3.12 -y
conda activate splitzip
pip install -r requirements.txt
```

If you already maintain a PyTorch CUDA environment, install the packages from
`requirements.txt` there instead of creating a new environment.

## Quick Start

Run a small codec smoke test with synthetic BF16 data:

```bash
python bench_codec_throughput.py \
  --device cuda:0 \
  --synthetic \
  --calibrate-on-input \
  --rows 1024 \
  --hidden-dim 4096 \
  --warmup 3 \
  --iters 10 \
  --repeats 2 \
  --output splitzip_smoke.json
```

The benchmark verifies bitwise lossless recovery before reporting compression
ratio and encode/decode throughput.

## Codec API

```python
import torch

from codec_gpu import ChunkLocalSplitZipGPU

x = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")

codec = ChunkLocalSplitZipGPU(device="cuda", chunk_size=1024)
coverage = codec.calibrate(x)

encoded = codec.encode(x)
decoded = codec.decode(encoded)

assert torch.equal(x.view(torch.int16), decoded.view(torch.int16))
print(coverage, encoded.compressed_bytes)
```

For paper-style experiments, calibrate the codebook on a separate calibration
set rather than on the benchmark tensor itself.

## Reproducing Paper Artifact Scripts

The paper protocol uses WikiText-2 with Qwen3-32B:

- codebook calibration: `wikitext/wikitext-2-raw-v1:train`;
- entropy/statistics evaluation: `wikitext/wikitext-2-raw-v1:test`;
- codec throughput tensor source: `wikitext/wikitext-2-raw-v1:test`.

Analyze BF16 KV exponent entropy:

```bash
python analyze_bf16_exponent_entropy_qwen32.py \
  --model Qwen/Qwen3-32B \
  --device-map auto \
  --max-prompts 4 \
  --output qwen3_32b_bf16_exponent_entropy.json
```

Run the public codec throughput benchmark:

```bash
python bench_codec_throughput.py \
  --device cuda:0 \
  --rows 65536 \
  --hidden-dim 4096 \
  --chunk-size 1024 \
  --warmup 10 \
  --iters 50 \
  --repeats 5 \
  --output splitzip_codec_throughput.json
```

For offline reproduction from saved BF16 tensors:

```bash
python bench_codec_throughput.py \
  --device cuda:0 \
  --input kv_tensor.pt \
  --calibration-input qwen32_wikitext2_train_kv.pt \
  --chunk-size 1024 \
  --output splitzip_codec_throughput.json
```

Use `--synthetic` and `--calibrate-on-input` only for kernel sanity checks; they
are not the paper protocol.

## Citation

If you find SplitZip useful in your research, please cite:

```bibtex
@article{guo2026splitzip,
  title         = {SplitZip: Ultra Fast Lossless KV Compression for Disaggregated LLM Serving},
  author        = {Guo, Yipin and Joshi, Siddharth},
  journal       = {arXiv preprint arXiv:2605.01708},
  year          = {2026},
  eprint        = {2605.01708},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC},
  url           = {https://arxiv.org/abs/2605.01708}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
