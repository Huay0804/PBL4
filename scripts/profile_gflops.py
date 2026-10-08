#!/usr/bin/env python3
"""Audit model-only inference arithmetic at batch 1, 512x1024.

Profiles saved Keras architecture configs (float32, optimizer excluded) and
fold-0 YOLO checkpoints. Writes new reports; never edits historical metrics.
Keras: TensorFlow registered graph FLOPs, plus explicit Einsum accounting.
YOLO: THOP x2 and PyTorch operator-level convolution/matmul FLOPs.
A multiply plus add counts as two operations. Backend coverage differs; inspect
per-operation reports rather than interpreting either total as hardware work.
Run in the project ML environment: python scripts/profile_gflops.py --all
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('U-Net', 'Modified U-Net', 'Modified NestNet', 'TransUNet', 'YOLO11-seg', 'YOLO26-seg')
H, W = 512, 1024


def checkpoint(name):
    paths = {
        'U-Net': 'runs/cv/fold_0/icpr_unet-resnet18/best.keras',
        'Modified U-Net': 'runs/cv/fold_0/icpr_munet-resnet18/best.keras',
        'Modified NestNet': 'runs/cv/fold_0/mod_nestnet/mod_nestnet20260504T145333/best.keras',
        'TransUNet': 'runs/transunet_runs/runs/cv/fold_0/transunet/transunet20260528T054805/best.keras',
        'YOLO11-seg': 'yolo_seg_models/runs/cv/fold_0/yolo11_seg/weights/best.pt',
        'YOLO26-seg': 'yolo_seg_models/runs/cv/fold_0/yolo26_seg/weights/best.pt',
    }
    return ROOT / paths[name]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def versions(names):
    result = {}
    for n in names:
        try:
            result[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def float32_config(obj):
    if isinstance(obj, dict):
        if obj.get('class_name') == 'DTypePolicy':
            obj['config']['name'] = 'float32'
        for v in obj.values():
            float32_config(v)
    elif isinstance(obj, list):
        for v in obj:
            float32_config(v)


def profile_keras(path):
    import tensorflow as tf
    import keras
    from tensorflow.python.framework import ops
    from tensorflow.python.profiler.internal import flops_registry  # registers counters
    sys.path.insert(0, str(ROOT / 'src'))
    import segmentation_models  # registers custom positional embedding layer

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    with zipfile.ZipFile(path) as z:
        config_bytes = z.read('config.json')
        config = json.loads(config_bytes)
    float32_config(config)
    config.pop('compile_config', None)
    model = keras.models.model_from_json(json.dumps(config))
    specs = [tf.TensorSpec((1, *x.shape[1:]), tf.float32) for x in model.inputs]
    assert all(s.shape[1:3] == (H, W) for s in specs)

    @tf.function
    def forward(*xs):
        return model(xs[0] if len(xs) == 1 else list(xs), training=False)

    concrete = forward.get_concrete_function(*specs)
    graph = concrete.graph
    options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    options['output'] = 'none'
    prof = tf.compat.v1.profiler.profile(graph, options=options)
    rows, missed, supplemental = [], Counter(), []
    counts = defaultdict(int)
    for op in graph.get_operations():
        try:
            value = ops.get_stats_for_node_def(graph, op.node_def, 'flops').value or 0
        except (ValueError, KeyError, TypeError) as e:
            raise RuntimeError(f'Cannot count {op.name} ({op.type}): {e}') from e
        if value:
            counts[op.type] += int(value)
            rows.append({'name': op.name, 'op': op.type, 'flops': int(value)})
        else:
            missed[op.type] += 1
        # Keras attention can use Einsum, which TF's registry does not count.
        if op.type == 'Einsum' and not value:
            eq = op.get_attr('equation').decode()
            lhs, rhs = eq.split('->')
            terms = lhs.split(',')
            if len(terms) != 2 or '.' in eq:
                raise ValueError(f'Unsupported Einsum equation: {eq}')
            dims = {}
            for term, tensor in zip(terms, op.inputs):
                shape = tensor.shape.as_list()
                if len(term) != len(shape) or None in shape:
                    raise ValueError(f'Unknown Einsum shape: {op.name}')
                for label, size in zip(term, shape):
                    if label in dims and dims[label] != size:
                        raise ValueError('Einsum broadcast needs explicit handling')
                    dims[label] = size
            contracted = set(''.join(terms)) - set(rhs)
            if not contracted:
                raise ValueError('Einsum without contraction needs a different rule')
            n = 2
            for size in dims.values():
                n *= size
            supplemental.append({'name': op.name, 'equation': eq, 'flops': n})
    assert sum(counts.values()) == prof.total_float_ops
    core = sum(v for k, v in counts.items() if k in
               {'Conv2D', 'Conv2DBackpropInput', 'DepthwiseConv2dNative', 'MatMul', 'BatchMatMul', 'BatchMatMulV2'})
    extra = sum(x['flops'] for x in supplemental)
    return {
        'backend': 'tensorflow', 'versions': versions(['tensorflow', 'keras']),
        'architecture_config_sha256': hashlib.sha256(config_bytes).hexdigest(),
        'architecture_source': 'saved checkpoint config; fresh weights; float32 policy',
        'params': model.count_params(), 'input_shapes': [s.shape.as_list() for s in specs],
        'output_shapes': [list(x.shape) for x in model.outputs], 'output_names': model.output_names,
        'tf_registered_gflops': prof.total_float_ops / 1e9,
        'supplemental_einsum_gflops': extra / 1e9,
        'gflops': (prof.total_float_ops + extra) / 1e9,
        'conv_matmul_gflops': (core + extra) / 1e9,
        'flops_by_op': dict(counts), 'uncounted_op_types': dict(missed),
        'operations': rows, 'supplemental_einsum_operations': supplemental,
        'graph_functions': list(graph.as_graph_def().library.function[i].signature.name
                                for i in range(len(graph.as_graph_def().library.function))),
        'limitations': 'Registered graph operations only; uncounted ops are listed. No backward pass, optimizer, prior detector, preprocessing, or argmax.'
    }


def profile_yolo(path):
    import torch
    import thop
    from ultralytics import YOLO
    from torch.utils.flop_counter import FlopCounterMode
    from thop.profile import register_hooks
    torch.set_num_threads(2)
    model = YOLO(str(path)).model.cpu().float().eval()
    x = torch.zeros(1, 3, H, W)
    shapes = {}
    transpose_checks = []
    def capture(name):
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                shapes[name] = list(output.shape)
            if isinstance(module, torch.nn.ConvTranspose2d):
                inp = inputs[0]
                # Scatter formulation: each input element contributes to every
                # output channel/kernel location (valid here: no padding).
                if any(module.padding) or any(module.output_padding):
                    raise ValueError('Transpose sanity check requires zero padding')
                kernel_area = module.kernel_size[0] * module.kernel_size[1]
                exact = 2 * inp.numel() * (module.out_channels // module.groups) * kernel_area
                thop_flops = 2 * output.numel() * (module.in_channels // module.groups) * kernel_area
                transpose_checks.append(dict(name=name, input_shape=list(inp.shape),
                    output_shape=list(output.shape), kernel=list(module.kernel_size),
                    stride=list(module.stride), scatter_flops=exact,
                    thop_output_based_flops=thop_flops))
        return hook
    handles = [m.register_forward_hook(capture(n)) for n, m in model.named_modules()
               if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))]
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        output = model(x)
    for h in handles:
        h.remove()
    core = fc.get_total_flops()
    core_ops = {str(k): int(v) for k, v in fc.get_flop_counts()['Global'].items()}
    missing = sorted({type(m).__name__ for m in model.modules() if type(m) not in register_hooks})
    with torch.no_grad():
        macs, thop_params, layer_info = thop.profile(model, inputs=(x,), verbose=False, ret_layer_info=True)
    return {
        'backend': 'pytorch', 'versions': versions(['torch','ultralytics','thop','ultralytics-thop']),
        'architecture_source': 'loaded fold-0 checkpoint, eval mode, float32, unfused',
        'params': sum(p.numel() for p in model.parameters()), 'input_shapes': [[1,3,H,W]],
        'gflops': 2 * macs / 1e9, 'thop_macs': macs,
        'conv_matmul_gflops': core / 1e9, 'torch_flops_by_op': core_ops,
        'thop_layer_counts': layer_info, 'conv_output_shapes': shapes,
        'transpose_convolution_checks': transpose_checks,
        'thop_source_sha256': {str(Path(thop.__file__).parent / f):
            sha256(Path(thop.__file__).parent / f)
            for f in ['profile.py', 'vision/basic_hooks.py', 'vision/calc_func.py']},
        'thop_unregistered_module_types': missing,
        'head_end2end': bool(getattr(model.model[-1], 'end2end', False)),
        'limitations': 'THOP x2 is a conventional estimate; it misses functional operations. PyTorch counter includes supported convolutions/matmuls/attention, not every elementwise op. No predict() preprocessing, NMS, or instance-mask rasterization.'
    }



def write_comparison(out_dir):
    historical_path = ROOT / 'runs/cv/figures/complexity.json'
    historical = json.loads(historical_path.read_text())
    rows = []
    for name in MODELS:
        data = json.loads((out_dir / (name.replace(' ', '_') + '.json')).read_text())
        key = {'U-Net':'ICPR U-Net', 'Modified U-Net':'ICPR Modified U-Net'}.get(name, name)
        old = historical[key]['gflops']
        rows.append(dict(model=name, historical_gflops=old,
                         reproduced_gflops=data['gflops'],
                         matches_reported_2dp=round(data['gflops'], 2) == old,
                         conv_matmul_gflops=data['conv_matmul_gflops']))
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  historical_file=str(historical_path.relative_to(ROOT)),
                  historical_sha256=sha256(historical_path), models=rows)
    (out_dir/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(rows, indent=2), flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--all', action='store_true')
    p.add_argument('--model', choices=MODELS)
    p.add_argument('--out-dir', type=Path, default=ROOT/'runs/cv/figures/gflops_audit')
    args = p.parse_args()
    if not args.all and not args.model:
        p.error('choose --all or --model')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.all:
        for name in MODELS:
            print('Profiling', name, flush=True)
            log = args.out_dir / (name.replace(' ', '_') + '.log')
            with log.open('w') as f:
                subprocess.run([sys.executable, __file__, '--model', name, '--out-dir', str(args.out_dir)],
                               stdout=f, stderr=subprocess.STDOUT, check=True)
        write_comparison(args.out_dir)
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    path = checkpoint(args.model)
    result = profile_keras(path) if path.suffix == '.keras' else profile_yolo(path)
    result.update(model=args.model, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=sha256(path),
                  script_sha256=sha256(Path(__file__)), batch=1,
                  convention='Multiply-add = 2 operations; see backend coverage and limitations.')
    target = args.out_dir / (args.model.replace(' ', '_') + '.json')
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(args.model, result['gflops'], 'GFLOPs; core', result['conv_matmul_gflops'])

if __name__ == '__main__':
    main()
