"""Profile parameters, MACs and CPU inference latency for every variant."""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.flops import model_profile  # noqa: E402
from src.models import MODEL_LABELS, MODEL_SPECS, build_model  # noqa: E402
from src.utils import configure_threads, ensure_dir, save_json  # noqa: E402


def benchmark_models(
    models: Dict[str, torch.nn.Module],
    batch_size: int,
    warmup: int = 20,
    rounds: int = 5,
    iters: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Measure all models in interleaved rounds.

    Measuring one model to completion and then the next gives the first model a
    cold-cache / frequency-ramp penalty and lets background drift bias whichever
    model happens to run during a slow interval.  Interleaving keeps the drift
    shared and the median over ``rounds * iters`` samples is robust to spikes.
    """
    xs = {
        name: torch.randn(batch_size, 3, 32, 32).contiguous(memory_format=torch.channels_last)
        for name in models
    }
    for model in models.values():
        model.eval()
    times: Dict[str, List[float]] = {name: [] for name in models}
    with torch.inference_mode():
        # Warm every model before timing any of them.
        for name, model in models.items():
            for _ in range(warmup):
                model(xs[name])
        for _ in range(rounds):
            for name, model in models.items():
                x = xs[name]
                for _ in range(iters):
                    start = time.perf_counter()
                    model(x)
                    times[name].append(time.perf_counter() - start)
    out: Dict[str, Dict[str, float]] = {}
    for name, values in times.items():
        times_ms = np.array(values) * 1000.0
        out[name] = {
            "batch_size": batch_size,
            "samples": int(times_ms.size),
            "latency_ms_mean": float(times_ms.mean()),
            "latency_ms_median": float(np.median(times_ms)),
            "latency_ms_p90": float(np.percentile(times_ms, 90)),
            "throughput_img_s": float(batch_size / (times_ms.mean() / 1000.0)),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--output", default="results/metrics/benchmark.json")
    args = p.parse_args()
    configure_threads(args.threads)

    models = {name: build_model(name).to(memory_format=torch.channels_last) for name in MODEL_SPECS}
    profiles = {name: model_profile(models[name]) for name in MODEL_SPECS}
    b1 = benchmark_models(models, 1, warmup=20, rounds=5, iters=10)
    b128 = benchmark_models(models, 128, warmup=10, rounds=5, iters=5)

    rows: List[Dict] = []
    for name in MODEL_SPECS:
        profile = profiles[name]
        row = {
            "model": name,
            "label": MODEL_LABELS[name],
            **profile,
            "size_mb": round(profile["params"] * 4 / 1e6, 3),
            "latency_b1": b1[name],
            "latency_b128": b128[name],
        }
        rows.append(row)
        print(
            f"{name:16s} params={row['params']:>8,} MACs={row['macs_m']:7.2f}M "
            f"b1={row['latency_b1']['latency_ms_median']:.2f}ms "
            f"b128={row['latency_b128']['throughput_img_s']:.0f} img/s"
        )
    ensure_dir(os.path.dirname(args.output))
    save_json(
        {
            "threads": args.threads,
            "torch": torch.__version__,
            "device": "cpu",
            "batch_note": "latency measured with torch.inference_mode, channels_last",
            "protocol": "models measured in interleaved rounds; b1 50 samples/model, b128 25 samples/model; median reported",
            "models": rows,
        },
        args.output,
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()