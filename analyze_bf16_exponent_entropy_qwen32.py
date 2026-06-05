import argparse
import json
from pathlib import Path
from typing import Iterable

import torch


DEFAULT_DATASET = "wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-2-raw-v1"
DEFAULT_ENTROPY_SPLIT = "test"


def iter_tensors(obj) -> Iterable[torch.Tensor]:
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


def update_hist(counts: torch.Tensor, tensor: torch.Tensor) -> None:
    flat = tensor.detach().to(torch.bfloat16).contiguous().view(torch.int16)
    exp = ((flat >> 7) & 0xFF).to(torch.long)
    counts += torch.bincount(exp.view(-1), minlength=256).cpu()


def summarize(counts: torch.Tensor) -> dict:
    total = int(counts.sum().item())
    if total == 0:
        raise RuntimeError("no BF16 values were analyzed")
    probs = counts.double() / total
    nonzero = counts > 0
    entropy = float(-(probs[nonzero] * torch.log2(probs[nonzero])).sum().item())
    order = torch.argsort(counts, descending=True)
    top = [
        {
            "rank": i + 1,
            "exponent": int(idx.item()),
            "count": int(counts[idx].item()),
            "frequency": float(counts[idx].item() / total),
        }
        for i, idx in enumerate(order[:16])
    ]
    return {
        "total_values": total,
        "unique_exponents": int(nonzero.sum().item()),
        "entropy_bits": entropy,
        "top8_coverage": float(counts[order[:8]].sum().item() / total),
        "top16_coverage": float(counts[order[:16]].sum().item() / total),
        "top16": top,
    }


def load_wikitext_texts(args) -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.entropy_split,
        cache_dir=args.dataset_cache_dir,
    )
    texts = []
    for row in dataset:
        text = " ".join(row["text"].split())
        if len(text) < args.min_text_chars:
            continue
        texts.append(text)
        if len(texts) >= args.max_prompts:
            break
    if not texts:
        raise RuntimeError(f"no usable text found in {args.dataset}/{args.dataset_config}:{args.entropy_split}")
    return texts


def load_prompts(args) -> tuple[list[str], dict]:
    if args.prompts is None:
        return load_wikitext_texts(args), {
            "text_source": "dataset",
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.entropy_split,
            "min_text_chars": args.min_text_chars,
        }
    path = args.prompts
    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return prompts[:args.max_prompts], {"text_source": "prompt_file", "path": str(path)}


def analyze_saved_tensors(paths: list[Path], cache_part: str) -> tuple[torch.Tensor, dict]:
    counts = torch.zeros(256, dtype=torch.long)
    tensors = 0
    for path in paths:
        obj = torch.load(path, map_location="cpu")
        for tensor in iter_tensors(obj):
            if tensor.dtype in {torch.float16, torch.bfloat16, torch.float32}:
                update_hist(counts, tensor)
                tensors += 1
    return counts, {"source": "saved_tensors", "files": [str(p) for p in paths], "tensors": tensors, "cache_part": cache_part}


def analyze_model(args) -> tuple[torch.Tensor, dict]:
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

    prompts, text_meta = load_prompts(args)
    counts = torch.zeros(256, dtype=torch.long)
    token_counts = []
    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, use_cache=True)
            token_counts.append(int(inputs["input_ids"].numel()))
            for key, value in iter_past_key_values(outputs.past_key_values):
                if args.cache_part in {"k", "both"}:
                    update_hist(counts, key)
                if args.cache_part in {"v", "both"}:
                    update_hist(counts, value)
    return counts, {
        "source": "model_forward",
        "model": args.model,
        "cache_part": args.cache_part,
        "prompts": len(prompts),
        "prompt_tokens": token_counts,
        **text_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze BF16 KV exponent entropy for Qwen3-32B.")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--input", type=Path, nargs="*", help="Optional saved tensor files; skips model loading.")
    parser.add_argument("--cache-part", choices=["k", "v", "both"], default="both")
    parser.add_argument("--prompts", type=Path, help="Optional prompt file. Default uses WikiText-2 test.")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--entropy-split", default=DEFAULT_ENTROPY_SPLIT)
    parser.add_argument("--dataset-cache-dir")
    parser.add_argument("--min-text-chars", type=int, default=80)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("qwen3_32b_bf16_exponent_entropy.json"))
    args = parser.parse_args()

    counts, meta = analyze_saved_tensors(args.input, args.cache_part) if args.input else analyze_model(args)
    result = {"metadata": meta, **summarize(counts), "histogram": counts.tolist()}
    args.output.write_text(json.dumps(result, indent=2))

    print(f"entropy_bits={result['entropy_bits']:.6f}")
    print(f"top8_coverage={result['top8_coverage'] * 100:.4f}%")
    print(f"top16_coverage={result['top16_coverage'] * 100:.4f}%")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
