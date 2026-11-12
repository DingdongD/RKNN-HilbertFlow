#!/usr/bin/env python3
"""Build the offline 3D demo with verified full-cloud FPS replay."""
from pathlib import Path
import csv
import json
import hashlib
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STORY = ROOT / 'outputs/rk3588/paper_experiments_2026-11-08/figures/story_suite'
STEPS, BITS, BUCKET = 256, 10, 256


def hilbert_codes(p):
    # Vectorized transcription of csrc/fps_curve.cpp::hilbert3d (Skilling).
    lo = p.min(0)
    span = max(float(np.ptp(p, axis=0).max()), 1e-9)
    x = np.floor((p - lo) / span * ((1 << BITS) - 1) + .5).astype(np.uint32)
    q = 1 << (BITS - 1)
    while q > 1:
        mask = q - 1
        for dim in range(3):
            hit = (x[:, dim] & q) != 0
            t = (x[:, 0] ^ x[:, dim]) & mask
            t[hit] = 0
            x[:, 0] ^= t
            x[:, dim] ^= t
            x[hit, 0] ^= mask
        q >>= 1
    x[:, 1] ^= x[:, 0]
    x[:, 2] ^= x[:, 1]
    t = np.zeros(len(p), np.uint32)
    q = 1 << (BITS - 1)
    while q > 1:
        t[(x[:, 2] & q) != 0] ^= q - 1
        q >>= 1
    x ^= t[:, None]
    code = np.zeros(len(p), np.uint32)
    for bit in range(BITS):
        for dim in range(3):
            code |= ((x[:, dim] >> (BITS - 1 - bit)) & 1) << (3 * BITS - 1 - (bit * 3 + dim))
    return code


def trace(p, view):
    """Kernel fine-bucket predicate; dense oracle verified at every iteration.

    Float64 reference replay on full scene, independent of RK3588 timing.
    It is not an instruction trace from the NEON implementation.
    """
    p = p.astype(np.float64)
    order = np.argsort(hilbert_codes(p), kind='stable')
    starts = np.arange(0, len(p), BUCKET)
    lo = np.minimum.reduceat(p[order], starts)
    hi = np.maximum.reduceat(p[order], starts)
    ids = np.empty(len(p), np.int32)
    ids[order] = np.arange(len(p)) // BUCKET
    nearest = np.full(len(p), np.inf)
    oracle = nearest.copy()
    selected = [0]
    nearest[0] = oracle[0] = -1
    frames, evaluated = [], 0
    for step in range(STEPS):
        current = selected[-1]
        maxima = np.maximum(np.maximum.reduceat(nearest[order], starts), 0)
        delta = np.maximum(0, np.maximum(lo - p[current], p[current] - hi))
        active = np.sum(delta * delta, axis=1) < maxima
        candidates = active[ids]
        # Dense distances are the verification oracle, excluded from replay counters.
        sq = np.sum((p - p[current]) ** 2, axis=1)
        oracle = np.minimum(oracle, sq)
        nearest[candidates] = np.minimum(nearest[candidates], sq[candidates])
        if not np.array_equal(nearest, oracle):
            raise AssertionError(f'unsafe pruning at round {step}')
        nxt = int(np.argmax(nearest))
        assert nxt == int(np.argmax(oracle))
        work = int(candidates.sum())
        evaluated += work
        frames.append({'center': current, 'next': nxt, 'active': np.flatnonzero(active).tolist(),
                       'evaluated': work, 'skipped': len(p) - work,
                       'cumulativePruned': 100 * (1 - evaluated / ((step + 1) * len(p)))})
        selected.append(nxt)
        nearest[nxt] = oracle[nxt] = -1
    return {'frames': frames, 'selected': selected, 'viewBuckets': ids[view].tolist(),
            'boxes': np.concatenate([lo, hi], axis=1).round(6).tolist(),
            'bits': BITS, 'bucketSize': BUCKET, 'exactVerified': True,
            'arithmetic': 'float64 reference replay',
            'source': 'csrc/fps_curve.cpp::hilbert3d and AABB fine-bucket update predicate'}


def load_scene(tag, folder, stem, manifest_name):
    folder = ROOT / folder
    mp = folder / manifest_name
    manifest = json.loads(mp.read_text())
    pp = folder / f'{stem}_positions.npy'
    p = np.load(pp)[0].T.astype(np.float64)
    room_path = Path(manifest['room'])
    raw = np.load(room_path)
    assert raw.shape == (len(p), 7)
    translation = np.median(p - raw[:, :3], axis=0)
    assert np.max(np.abs(p - raw[:, :3] - translation)) < 1e-5
    view = np.sort(np.random.default_rng(42).choice(len(p), min(24000, len(p)), replace=False))
    fps = trace(p, view)
    rgb = np.clip(raw[view, 3:6], 0, 255).astype(np.uint8)
    result = {'tag': tag, 'name': room_path.stem.replace('_', ' '), 'count': len(p),
              'xyz': p[view].round(6).tolist(), 'rgb': rgb.tolist(),
              'groundTruth': raw[view, 6].astype(int).tolist(),
              'bounds': [p.min(0).tolist(), p.max(0).tolist()],
              'fps': fps, 'selectedXYZ': p[fps['selected']].round(6).tolist(),
              'positionSource': str(pp.relative_to(ROOT)), 'sceneSource': str(room_path),
              'manifestSource': str(mp.relative_to(ROOT)), 'models': {}}
    if tag in ('93K', '202K', '503K'):
        level = np.arange(len(p))
        hierarchy = []
        for i in range(4):
            indices = np.load(folder / f'{stem}_sample{i}.npy').reshape(-1).astype(int)
            level = level[indices]
            display = level[np.linspace(0, len(level)-1, min(10000, len(level)), dtype=int)]
            hierarchy.append({'count': len(level), 'xyz': p[display].round(6).tolist()})
        result['hierarchy'] = hierarchy
        model_paths = [
            ('PointNeXt-L', 'outputs/rk3588/pointnext_l_s3dis_grouped/pointnext_s3dis_l_reference.npy'),
            ('PointVector-XL', 'outputs/rk3588/pointvector_xl_s3dis_grouped_q512/pointvector_s3dis_xl_reference.npy')]
        if tag != '93K':
            model_paths = [('PointNeXt-L', f'outputs/rk3588/s3dis_{tag.lower()}_acceptance/pointnext_l_reference/pointnext_s3dis_l_reference.npy')]
        for name, path in model_paths:
            logits = np.load(ROOT / path, mmap_mode='r')
            assert logits.shape == (1, 13, len(p)) and np.isfinite(logits).all()
            labels = np.argmax(logits[0], axis=0)
            result['models'][name] = {'prediction': labels[view].astype(int).tolist(),
                                     'source': path, 'backend': 'stored CUDA checkpoint reference',
                                     'sha256': hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
    print(f'{tag}: {len(p):,} points, {STEPS} verified rounds, {len(fps["boxes"])} buckets', flush=True)
    return result


def main():
    scenes = [
        load_scene('93K', 'outputs/rk3588/pointnext_s3dis_external_indices', 'pointnext_s3dis_n93133', 'pointnext_s3dis_n93133_external_indices.manifest.json'),
        load_scene('202K', 'outputs/rk3588/s3dis_202k_acceptance/input', 'pointnext_s3dis_n201841', 'pointnext_s3dis_n201841_input_manifest.json'),
        load_scene('503K', 'outputs/rk3588/s3dis_503k_acceptance/input', 'pointnext_s3dis_n502947', 'pointnext_s3dis_n502947_input_manifest.json')]
    sources = ['fps_speedup_memory_pruning_speedup_data.csv', 'model_level_speedup_comparison_data.csv']
    with (STORY / sources[0]).open() as f:
        fps = [r for r in csv.DictReader(f) if r['method'] in ('hilbert', 'vanilla')]
    with (STORY / sources[1]).open() as f:
        models = [r for r in csv.DictReader(f) if r['model'] in ('PointNeXt-L', 'PointVector-XL') and r['method'] in ('vanilla','hilberflow_full')]
    locked = ROOT/'outputs/rk3588/paper_experiments_2026-11-11/figures/story_suite/model_level_speedup_comparison_final_locked_data.csv'
    with locked.open() as f:
        for r in csv.DictReader(f):
            if r['model'] == 'PointNeXt-L' and int(r['points']) in (201841, 502947):
                models.append({'model': r['model'], 'points': r['points'], 'method': 'hilberflow_full',
                               'total_ms': r['hilbertflow_ms'], 'baseline_ms': r['b0_ms'],
                               'speedup': r['speedup'], 'clock_scope': 'fixed-clock paired sweep'})
    payload = {'scenes': scenes, 'fpsMetrics': fps, 'modelMetrics': models,
               'sources': [str((STORY / s).relative_to(ROOT)) for s in sources] + [str(locked.relative_to(ROOT))]}
    data = json.dumps(payload, separators=(',', ':'), ensure_ascii=True).replace('</', '<\\/')
    html = (HERE/'index.template.html').read_text().replace('__EMBEDDED_DATA__', data)
    (HERE/'index.html').write_text(html)
    audit = {'scene_rounds': {s['tag']: len(s['fps']['frames']) for s in scenes},
             'all_rounds_match_dense': True, 'replay_arithmetic': 'float64',
             'trace_is_rk3588_instruction_trace': False,
             'prediction_backend': 'stored CUDA checkpoint reference',
             'timing_sources': payload['sources']}
    (HERE/'replay_audit.json').write_text(json.dumps(audit, indent=2))
    with zipfile.ZipFile(ROOT/'demo/HilbertFlow_RK3588_Demo.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in ['index.html','index.template.html','build_demo.py','build_replay.py','README.md','replay_audit.json']:
            z.write(HERE/name, f'HilbertFlow_RK3588_Demo/{name}')
        for s in sources:
            z.write(STORY/s, f'HilbertFlow_RK3588_Demo/data/{s}')
        z.write(locked, f'HilbertFlow_RK3588_Demo/data/{locked.name}')
    print(f'Built {len(html):,} bytes; ZIP refreshed.', flush=True)


if __name__ == '__main__':
    main()
