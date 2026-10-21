#!/usr/bin/env python3
"""Continue AugValBench six-condition stress runs without redoing completed stages."""
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / 'AugValBench_v1'
PY = str(Path.home() / 'miniconda3/envs/rec/bin/python')
MODELS = [
    'microsoft/Phi-3.5-mini-instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
]
DATASETS = ['All_Beauty', 'Gift_Cards', 'MovieLens_1M']
SOURCE_DIRS = {
    'All_Beauty': ROOT / 'runs/All_Beauty/data',
    'Gift_Cards': ROOT / 'runs/Gift_Cards/data',
    'MovieLens_1M': ROOT / 'baseline_runs/MovieLens_1M/data',
}
CONDS = ['base', 'clean', 'conflict', 'irrelevant', 'partial', 'redundant']
PERTURB = ['conflict', 'irrelevant', 'partial', 'redundant']


def say(*args):
    print(*args, flush=True)


def expected_cases(dataset):
    source = SOURCE_DIRS[dataset]
    data = ROOT / 'benchmark_runs' / dataset / 'data'
    data.mkdir(parents=True, exist_ok=True)
    mapping = {
        'formal/eval_test.jsonl': 'eval_cases.jsonl',
        'train_sequences.jsonl': 'train_sequences.jsonl',
        'item_popularity.json': 'item_popularity.json',
        'item_meta.json': 'item_meta.json',
    }
    for srcname, dstname in mapping.items():
        src = source / srcname
        dst = data / dstname
        if not src.is_file():
            raise RuntimeError(f'Missing source data: {src}')
        if not dst.exists():
            shutil.copy2(src, dst)
        else:
            import filecmp
            if not filecmp.cmp(src, dst, shallow=False):
                raise RuntimeError(
                    f'Data mismatch: {src} != {dst}\n'
                    'STOP: benchmark inputs may not match previous scores.'
                )
    with (data / 'eval_cases.jsonl').open(encoding='utf-8') as f:
        ids = [int(json.loads(line)['case_id']) for line in f if line.strip()]
    if len(ids) != len(set(ids)) or not ids:
        raise RuntimeError(f'Bad test case IDs in {data}')
    return data, set(ids)


def read_pair(folder, condition, ids):
    sp = folder / 'rerank_scores.csv'
    mp = folder / 'case_metrics.csv'
    if not sp.is_file() or not mp.is_file():
        return None
    try:
        s = pd.read_csv(sp)
        m = pd.read_csv(mp)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    required_s = {'case_id', 'condition', 'item_id', 'score', 'is_target'}
    required_m = {'case_id', 'condition'}
    if not required_s.issubset(s.columns) or not required_m.issubset(m.columns):
        return None
    s = s.loc[s['condition'].astype(str) == condition].copy()
    m = m.loc[m['condition'].astype(str) == condition].copy()
    n = len(ids)
    if len(s) != 20 * n or len(m) != n:
        return None
    try:
        s['case_id'] = s['case_id'].astype(int)
        m['case_id'] = m['case_id'].astype(int)
    except (ValueError, TypeError):
        return None
    if set(s['case_id']) != ids or set(m['case_id']) != ids:
        return None
    if not s.groupby('case_id').size().eq(20).all():
        return None
    if s.duplicated(['case_id', 'item_id']).any() or m.duplicated(['case_id']).any():
        return None
    vals = pd.to_numeric(s['score'], errors='coerce').to_numpy(dtype='float64')
    if not np.isfinite(vals).all():
        return None
    return s, m


def validate_alignment(a, b, a_name, b_name):
    x = a.copy()
    y = b.copy()
    x['item_id'] = x['item_id'].astype(str)
    y['item_id'] = y['item_id'].astype(str)
    x = x.set_index(['case_id', 'item_id']).sort_index()
    y = y.set_index(['case_id', 'item_id']).sort_index()
    if not x.index.equals(y.index):
        raise RuntimeError(f'Mismatched cases/items: {a_name} vs {b_name}')
    if not x['is_target'].astype(str).equals(y['is_target'].astype(str)):
        raise RuntimeError(f'Mismatched target labels: {a_name} vs {b_name}')


def save_csv_atomic(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def env_for(dataset, model, data, result, conditions):
    env = os.environ.copy()
    env.update({
        'HF_HOME': '/data/users/zhjia/cache/huggingface',
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
        'CUDA_VISIBLE_DEVICES': '0',
        'PYTHONUNBUFFERED': '1',
        'AUGVAL_DATASET_NAME': dataset,
        'AUGVAL_AMAZON_CATEGORY': dataset,
        'AUGVAL_MODEL_NAME': model,
        'AUGVAL_DATA_DIR': str(data),
        'AUGVAL_RESULT_DIR': str(result),
        'AUGVAL_CONDITIONS': conditions,
    })
    return env


def run(command, env):
    say('>>>', ' '.join(map(str, command)))
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def collect_source(pair, dataset, model, data, ids):
    source = pair / 'source'
    source.mkdir(parents=True, exist_ok=True)
    src_parts = {}
    for cond in ['base', 'clean']:
        folders = [
            source,
            pair / f'source_{cond}_only',
            pair / 'source_parts' / cond,
        ]
        selected = None
        for folder in folders:
            selected = read_pair(folder, cond, ids)
            if selected is not None:
                say('REUSE source:', cond, folder)
                break
        if selected is None:
            out = pair / 'source_parts' / cond
            out.mkdir(parents=True, exist_ok=True)
            say('RUN missing source condition:', dataset, model, cond)
            run([PY, '-u', '-m', 'src.run_rerank'],
                env_for(dataset, model, data, out, cond))
            selected = read_pair(out, cond, ids)
            if selected is None:
                raise RuntimeError(f'Invalid {cond} scores or metrics: {out}')
        src_parts[cond] = selected

    validate_alignment(src_parts['base'][0], src_parts['clean'][0], 'base', 'clean')
    score_cols = list(src_parts['base'][0].columns)
    metric_cols = list(src_parts['base'][1].columns)
    for condition in ('base', 'clean'):
        scores, metrics = src_parts[condition]
        if set(scores.columns) != set(score_cols) or set(metrics.columns) != set(metric_cols):
            raise RuntimeError('base/clean CSV schemas differ')
    full_scores = pd.concat([src_parts[x][0][score_cols] for x in ('base', 'clean')], ignore_index=True)
    full_metrics = pd.concat([src_parts[x][1][metric_cols] for x in ('base', 'clean')], ignore_index=True)
    for name, df in [('rerank_scores.csv', full_scores), ('case_metrics.csv', full_metrics)]:
        dest = source / name
        if dest.exists() and not (source / (name + '.before_resume.bak')).exists():
            shutil.copy2(dest, source / (name + '.before_resume.bak'))
        save_csv_atomic(df, dest)
    say('SOURCE VERIFIED:', dataset, model, len(full_scores), 'score rows', len(full_metrics), 'metric rows')
    return source


def check_conditions(folder, conditions, ids):
    result = {}
    for c in conditions:
        val = read_pair(folder, c, ids)
        if val is None:
            return None
        result[c] = val
    for cond in conditions[1:]:
        validate_alignment(result[conditions[0]][0], result[cond][0], conditions[0], cond)
    return result


def complete(pair, ids):
    final = pair / 'benchmark' / 'results'
    if not (final / 'benchmark_summary.csv').is_file():
        return False
    p = final / 'summary_by_condition.csv'
    if not p.is_file():
        return False
    try:
        summ = pd.read_csv(p)
        if set(summ['condition'].astype(str)) != set(CONDS):
            return False
        if 'n' in summ.columns and not summ['n'].astype(int).eq(len(ids)).all():
            return False
        return check_conditions(final, CONDS, ids) is not None
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False


def do_pair(dataset, model):
    slug = model.replace('/', '__')
    pair = ROOT / 'cross_model_runs' / dataset / slug
    data, ids = expected_cases(dataset)
    say('\n' + '=' * 72)
    say('PAIR:', dataset, model, '|', len(ids), 'cases')
    say('=' * 72)
    if complete(pair, ids):
        say('SKIP COMPLETED:', dataset, model)
        return

    source = collect_source(pair, dataset, model, data, ids)
    perturb = pair / 'benchmark' / 'perturb'
    perturb.mkdir(parents=True, exist_ok=True)
    if check_conditions(perturb, PERTURB, ids) is None:
        say('RUN four perturbations:', dataset, model)
        run([PY, '-u', '-m', 'src.run_rerank'],
            env_for(dataset, model, data, perturb, ','.join(PERTURB)))
        if check_conditions(perturb, PERTURB, ids) is None:
            raise RuntimeError(f'Invalid perturbation scores or metrics: {perturb}')
    else:
        say('REUSE four perturbations:', dataset, model)

    final = pair / 'benchmark' / 'results'
    final.mkdir(parents=True, exist_ok=True)
    env = env_for(dataset, model, data, final, ','.join(CONDS))
    run([
        PY, '-u', '-m', 'src.benchmark.assemble',
        '--source-result', str(source),
        '--perturb-result', str(perturb),
        '--benchmark-result', str(final),
        '--benchmark-data', str(data),
    ], env)
    run([PY, '-u', '-m', 'src.analyze_gain'], env)
    run([PY, '-u', '-m', 'src.benchmark.analyze_benchmark'], env)
    if not complete(pair, ids):
        raise RuntimeError(f'Final results are not complete: {final}')
    say('PAIR DONE:', dataset, model)


def main():
    if not Path(PY).is_file():
        raise RuntimeError(f'Python executable missing: {PY}')
    os.chdir(ROOT)
    for model in MODELS:
        for dataset in DATASETS:
            do_pair(dataset, model)
    say('\nALL SIX CROSS-FAMILY BENCHMARKS COMPLETE')


if __name__ == '__main__':
    main()
