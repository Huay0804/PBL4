#!/usr/bin/env python3
"""Measure inference latency and parameter counts of the six segmentation models.

Latency depends only on architecture (not trained weights), so each model is
built fresh and a forward pass is timed. Each model runs in its **own
subprocess** so TensorFlow and Torch never fight over the GPU and memory does
not accumulate across models.

Protocol (report these alongside the numbers — latency is hardware-dependent):
  * one image of shape 512x1024 (+ a 33-channel bbox-prior tensor for the
    prior-gated models), batch size 1;
  * dense Keras models timed with a graph-mode ``tf.function`` forward pass;
  * YOLO-seg timed via ``model.predict`` (reports both wall-clock and
    Ultralytics' own inference-only speed);
  * ``--warmup`` untimed runs, then ``--repeat`` timed runs; mean +/- std in ms;
  * Modified NestNet is built with the SAME deep-supervision head as training
    (read from project_presets), which is what determines its param count.

NOT measured here: GFLOPs (needs a separate profiler) — params + latency only.

Usage::

    python scripts/benchmark_latency.py --all
    python scripts/benchmark_latency.py --all --warmup 15 --repeat 50
    python scripts/benchmark_latency.py --model "Modified NestNet"   # one model
    python scripts/benchmark_latency.py --all --out runs/cv/figures/latency_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"

H, W, C = 512, 1024, 33
KERAS_MODELS = ["U-Net", "Modified U-Net", "Modified NestNet", "TransUNet"]
GATED = {"Modified U-Net", "Modified NestNet"}

# YOLO weights: prefer the CV fold-0 checkpoints, fall back to the app's copies.
def _yolo_paths():
    out = {}
    for tag, run in [("YOLO11-seg", "yolo11_seg"), ("YOLO26-seg", "yolo26_seg")]:
        cands = [
            REPO / "yolo_seg_models" / "runs" / "cv" / "fold_0" / run / "weights" / "best.pt",
            REPO / "apps" / "tooth_charting_assistant" / "models" / f"{run}.pt",
        ]
        hit = next((p for p in cands if p.exists()), None)
        if hit:
            out[tag] = str(hit)
    return out


# ---------------------------------------------------------------- Keras backend
def _nestnet_ds_index():
    """Deep-supervision output index used at TRAIN time (mirrors train.py)."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        from project_presets import get_segmentation_preset
        p = get_segmentation_preset("mod_nestnet")
        head = p.get("ds_train_head", "index")
        if head == "all":
            return None  # multi-head — not the deployed config
        if head == "last":
            return 3
        return int(p.get("ds_train_output_index", 2))
    except Exception:
        return 2  # documented default for this project


def bench_keras(name, warmup, repeat, mixed):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    sys.path.insert(0, str(SRC))
    import numpy as np
    import tensorflow as tf

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    if mixed:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    import segmentation_models as M

    if name == "U-Net":
        model = M.ICPRUnet(input_shape=(H, W, 3), classes=C, activation="softmax")
    elif name == "Modified U-Net":
        model = M.ICPRModifiedUnet(input_shape=(H, W, 3), classes=C, activation="softmax", bb_channels=C)
    elif name == "Modified NestNet":
        model = M.ModifiedNestnet(input_shape=(H, W, 3), classes=C, activation="softmax",
                                  bb_channels=C, deep_supervision=False,
                                  deep_supervision_output_index=_nestnet_ds_index())
    elif name == "TransUNet":
        model = M.TransUNet(input_shape=(H, W, 3), classes=C, activation="softmax")
    else:
        raise ValueError(name)

    params = model.count_params()
    gated = name in GATED
    # pre-placed tensors for the forward-only timing
    img_t = tf.constant(np.zeros((1, H, W, 3), np.float32))
    prior_t = tf.constant(np.zeros((1, H, W, C), np.float32))
    fwd_inp = (img_t, prior_t) if gated else img_t
    # raw uint8 image for the end-to-end timing (incl. preprocess + host<->device)
    raw = np.zeros((H, W, 3), np.uint8)
    prior_raw = np.zeros((H, W, C), np.float32)  # ready prior; detector/prior generation excluded

    @tf.function
    def fwd(x):
        return model(x, training=False)

    def first(y):
        return y[-1] if isinstance(y, (list, tuple)) else y

    def sync(y):
        for t in (y if isinstance(y, (list, tuple)) else [y]):
            t.numpy()

    def forward_only():
        sync(fwd(fwd_inp))

    def end_to_end():
        # preprocess: uint8 -> float, normalize, batch, to-device
        x = (raw.astype(np.float32) / 255.0)[None, ...]
        xt = tf.constant(x)
        inp = (xt, tf.constant(prior_raw[None])) if gated else xt
        y = fwd(inp)
        # postprocess: argmax over 33 classes -> label map, copy to host
        lab = tf.argmax(first(y), axis=-1)
        lab.numpy()

    def timeit(fn):
        for _ in range(warmup):
            fn()
        ts = []
        for _ in range(repeat):
            s = time.perf_counter(); fn(); ts.append((time.perf_counter() - s) * 1000.0)
        ts = np.array(ts)
        return round(float(ts.mean()), 1), round(float(ts.std()), 1)

    fwd_mean, fwd_std = timeit(forward_only)
    e2e_mean, e2e_std = timeit(end_to_end)
    gpu = tf.config.list_physical_devices("GPU")
    return {
        "backend": "keras",
        "params_M": round(params / 1e6, 3),
        "forward_ms_mean": fwd_mean, "forward_ms_std": fwd_std,
        "end_to_end_ms_mean": e2e_mean, "end_to_end_ms_std": e2e_std,
        "precision": "mixed_float16" if mixed else "float32",
        "device": "GPU" if gpu else "CPU",
    }


# ---------------------------------------------------------------- YOLO backend
def bench_yolo(name, path, warmup, repeat, imgsz):
    import numpy as np
    import torch
    from ultralytics import YOLO

    m = YOLO(path)
    params = sum(p.numel() for p in m.model.parameters()) / 1e6
    dev = 0 if torch.cuda.is_available() else "cpu"
    img = np.zeros((H, W, 3), np.uint8)
    for _ in range(warmup):
        m.predict(img, imgsz=imgsz, device=dev, verbose=False)
    wall, infer = [], []
    for _ in range(repeat):
        s = time.perf_counter()
        r = m.predict(img, imgsz=imgsz, device=dev, verbose=False)
        wall.append((time.perf_counter() - s) * 1000.0)
        infer.append(float(r[0].speed["inference"]))
    wall, infer = np.array(wall), np.array(infer)
    return {
        "backend": "ultralytics",
        "params_M": round(params, 3),
        # forward-only: Ultralytics' inference stage (torch.cuda.synchronize'd), comparable to Keras forward_ms
        "forward_ms_mean": round(float(infer.mean()), 1),
        "forward_ms_std": round(float(infer.std()), 1),
        # end-to-end: full predict() wall-clock = preprocess + inference + postprocess
        # (still excludes the instance->33-class rasterization the pipeline does afterwards)
        "end_to_end_ms_mean": round(float(wall.mean()), 1),
        "end_to_end_ms_std": round(float(wall.std()), 1),
        "imgsz": imgsz,
        "device": "GPU" if torch.cuda.is_available() else "CPU",
    }


def bench_one(name, warmup, repeat, mixed, imgsz):
    yolo = _yolo_paths()
    if name in yolo:
        return bench_yolo(name, yolo[name], warmup, repeat, imgsz)
    if name in KERAS_MODELS:
        return bench_keras(name, warmup, repeat, mixed)
    raise SystemExit(f"Unknown model: {name}")


# ---------------------------------------------------------------- orchestration
def _device_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        return out.splitlines()[0] if out else "CPU"
    except Exception:
        return "unknown"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="Benchmark a single model (runs in this process).")
    ap.add_argument("--all", action="store_true", help="Benchmark all six (one subprocess each).")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=1024, help="YOLO inference size.")
    ap.add_argument("--mixed", action="store_true", help="Use mixed_float16 for Keras models.")
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "cv" / "figures" / "latency_benchmark.json")
    return ap.parse_args()


def main():
    args = parse_args()

    # Single-model mode: do the work and emit a parseable result line.
    if args.model:
        res = bench_one(args.model, args.warmup, args.repeat, args.mixed, args.imgsz)
        print("__RESULT__" + json.dumps({args.model: res}))
        return

    if not args.all:
        raise SystemExit("Pass --all or --model NAME. See --help.")

    order = KERAS_MODELS + list(_yolo_paths())
    results = {}
    for name in order:
        print(f"\n>>> benchmarking {name} ...", flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--model", name,
               "--warmup", str(args.warmup), "--repeat", str(args.repeat),
               "--imgsz", str(args.imgsz)]
        if args.mixed:
            cmd.append("--mixed")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("__RESULT__")), None)
        if line:
            results.update(json.loads(line[len("__RESULT__"):]))
            r = results[name]
            print(f"    {name}: {r['params_M']}M params | forward {r['forward_ms_mean']} +/- {r['forward_ms_std']} ms"
                  f" | end-to-end {r['end_to_end_ms_mean']} +/- {r['end_to_end_ms_std']} ms")
        else:
            tail = (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1]
            results[name] = {"error": tail[:200]}
            print(f"    {name}: FAILED -> {tail[:160]}")

    payload = {
        "metadata": {
            "device": _device_name(),
            "warmup": args.warmup, "repeat": args.repeat,
            "input": f"{H}x{W}", "batch": 1, "imgsz_yolo": args.imgsz,
            "note": "Latency is hardware-dependent; report the device. "
                    "forward_ms = network forward only, GPU-synchronized (Keras: tf.function + .numpy(); "
                    "YOLO: Ultralytics' inference stage) — comparable across backends. "
                    "end_to_end_ms = full pipeline: Keras = preprocess(normalize)+forward+argmax; "
                    "YOLO = predict() (preprocess+inference+postprocess). Both end-to-end numbers "
                    "EXCLUDE prior/detector generation (gated models) and the instance->semantic "
                    "rasterization (YOLO). Params are architecture-exact.",
        },
        "models": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
