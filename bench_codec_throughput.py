import argparse
import json
import time
from pathlib import Path
from statistics import mean, stdev

import torch

from codec_gpu import ChunkLocalSplitZipGPU


DEFAULT_DATASET = "wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-2-raw-v1"
DEFAULT_CALIBRATION_SPLIT = "train"
DEFAULT_BENCHMARK_SPLIT = "test"


def iter_tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from iter_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from iter_tensors(value)


def iter_past_key_values(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None:
                key = getattr(layer, "key_cache", None)
            if value is None:
                value = getattr(layer, "value_cache", None)
            if key is not None and value is not None:
                yield key, value
        return
    for layer in past_key_values:
        if isinstance(layer, (tuple, list)) and len(layer) >= 2:
            yield layer[0], layer[1]


def load_tensor(path: Path, device: str, numel: int | None) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    pieces = []
    for tensor in iter_tensors(obj):
        if tensor.dtype in {torch.float16, torch.bfloat16, torch.float32}:
            pieces.append(tensor.reshape(-1).to(torch.bfloat16))
    if not pieces:
        raise RuntimeError(f"no floating-point tensors found in {path}")
    flat = torch.cat(pieces)
    if numel is not None:
        if flat.numel() < numel:
            repeat = (numel + flat.numel() - 1) // flat.numel()
            flat = flat.repeat(repeat)
        flat = flat[:numel]
    return flat.contiguous().to(device)


def load_wikitext_texts(args, split: str, max_prompts: int) -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=split,
        cache_dir=args.dataset_cache_dir,
    )
    texts = []
    for row in dataset:
        text = " ".join(row["text"].split())
        if len(text) < args.min_text_chars:
            continue
        texts.append(text)
        if len(texts) >= max_prompts:
            break
    if not texts:
        raise RuntimeError(f"no usable text found in {args.dataset}/{args.dataset_config}:{split}")
    return texts


def collect_wikitext_kv_tensor(args, split: str, max_prompts: int, target_values: int) -> torch.Tensor:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    model_kwargs = dict(
        device_map=args.device_map,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, **model_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, **model_kwargs)
    model.eval()

    pieces = []
    values = 0
    texts = load_wikitext_texts(args, split, max_prompts)
    with torch.inference_mode():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            outputs = model(**inputs, use_cache=True)
            for key, value in iter_past_key_values(outputs.past_key_values):
                for tensor in (key, value):
                    flat = tensor.detach().to(torch.bfloat16).reshape(-1).cpu()
                    remaining = target_values - values
                    if remaining <= 0:
                        break
                    pieces.append(flat[:remaining])
                    values += min(flat.numel(), remaining)
                if values >= target_values:
                    break
            if values >= target_values:
                break
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    if not pieces:
        raise RuntimeError("no KV values were collected")
    flat = torch.cat(pieces).contiguous()
    if flat.numel() < target_values:
        repeat = (target_values + flat.numel() - 1) // flat.numel()
        flat = flat.repeat(repeat)[:target_values].contiguous()
    return flat.to(args.device)


def target_numel(args) -> int:
    return args.rows * args.hidden_dim if args.rows and args.hidden_dim else args.numel


def make_tensor(args) -> torch.Tensor:
    numel = target_numel(args)
    if args.input:
        return load_tensor(args.input, args.device, numel)
    if not args.synthetic:
        return collect_wikitext_kv_tensor(args, args.benchmark_split, args.max_benchmark_prompts, numel)
    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed)
    return torch.randn(numel, dtype=torch.bfloat16, device=args.device, generator=gen)


def make_calibration_tensor(args, benchmark_tensor: torch.Tensor) -> tuple[torch.Tensor, dict]:
    if args.calibration_input:
        return load_tensor(args.calibration_input, args.device, args.max_calibration_values), {
            "calibration_source": "saved_tensor",
            "path": str(args.calibration_input),
        }
    if args.calibrate_on_input:
        return benchmark_tensor, {"calibration_source": "benchmark_input"}
    return collect_wikitext_kv_tensor(
        args,
        args.calibration_split,
        args.max_calibration_prompts,
        args.max_calibration_values,
    ), {
        "calibration_source": "model_forward",
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.calibration_split,
        "max_prompts": args.max_calibration_prompts,
        "max_values": args.max_calibration_values,
    }


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def bench(fn, device: str, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / iters


def summarize(values: list[float]) -> dict:
    return {
        "mean": mean(values),
        "min": min(values),
        "max": max(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SplitZip codec_gpu encode/decode throughput.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--numel", type=int, default=65536 * 4096)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--input", type=Path, help="Optional saved BF16 tensor source. Default uses WikiText-2 test KV.")
    parser.add_argument("--synthetic", action="store_true", help="Smoke-test shortcut; paper throughput uses WikiText-2 test KV.")
    parser.add_argument("--calibration-input", type=Path, help="Optional saved BF16 tensor for codebook calibration.")
    parser.add_argument("--calibrate-on-input", action="store_true", help="Smoke-test shortcut; paper runs calibrate on WikiText-2 train.")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--calibration-split", default=DEFAULT_CALIBRATION_SPLIT)
    parser.add_argument("--benchmark-split", default=DEFAULT_BENCHMARK_SPLIT)
    parser.add_argument("--dataset-cache-dir")
    parser.add_argument("--max-calibration-prompts", type=int, default=4)
    parser.add_argument("--max-benchmark-prompts", type=int, default=4)
    parser.add_argument("--max-calibration-values", type=int, default=4_194_304)
    parser.add_argument("--min-text-chars", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("splitzip_codec_throughput.json"))
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    x = make_tensor(args).contiguous()
    codec = ChunkLocalSplitZipGPU(device=args.device, chunk_size=args.chunk_size)
    calibration_tensor, calibration_meta = make_calibration_tensor(args, x)
    coverage = codec.calibrate(calibration_tensor)
    if calibration_tensor is not x:
        del calibration_tensor
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    encoded = codec.encode(x)
    decoded = codec.decode(encoded)
    if not torch.equal(x.view(torch.int16), decoded.view(torch.int16)):
        raise RuntimeError("lossless round trip failed")

    raw_bytes = x.numel() * 2
    ratio = raw_bytes / encoded.compressed_bytes
    enc_times = [bench(lambda: codec.encode(x), args.device, args.warmup, args.iters) for _ in range(args.repeats)]
    dec_times = [bench(lambda: codec.decode(encoded), args.device, args.warmup, args.iters) for _ in range(args.repeats)]
    enc_gbs = [raw_bytes / t / 1e9 for t in enc_times]
    dec_gbs = [raw_bytes / t / 1e9 for t in dec_times]

    result = {
        "device": args.device,
        "shape": list(x.shape),
        "numel": int(x.numel()),
        "chunk_size": args.chunk_size,
        "raw_bytes": raw_bytes,
        "compressed_bytes": encoded.compressed_bytes,
        "compression_ratio": ratio,
        "top16_coverage": coverage,
        "n_escapes": encoded.n_esc,
        "escape_rate": encoded.n_esc / x.numel(),
        "calibration": calibration_meta,
        "benchmark_source": {
            "source": "saved_tensor" if args.input else ("synthetic" if args.synthetic else "model_forward"),
            "path": str(args.input) if args.input else None,
            "model": None if args.input or args.synthetic else args.model,
            "dataset": None if args.input or args.synthetic else args.dataset,
            "dataset_config": None if args.input or args.synthetic else args.dataset_config,
            "split": None if args.input or args.synthetic else args.benchmark_split,
            "max_prompts": None if args.input or args.synthetic else args.max_benchmark_prompts,
        },
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "encode_seconds": summarize(enc_times),
        "decode_seconds": summarize(dec_times),
        "encode_gbs": summarize(enc_gbs),
        "decode_gbs": summarize(dec_gbs),
    }
    args.output.write_text(json.dumps(result, indent=2))

    print(f"ratio={ratio:.6f} coverage={coverage * 100:.4f}% escapes={encoded.n_esc}")
    print(f"encode_gbs={result['encode_gbs']['mean']:.3f} +/- {result['encode_gbs']['std']:.3f}")
    print(f"decode_gbs={result['decode_gbs']['mean']:.3f} +/- {result['decode_gbs']['std']:.3f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
