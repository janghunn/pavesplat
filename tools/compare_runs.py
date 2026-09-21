#!/usr/bin/env python
"""Supplement section 6 / Figure 1: compare local W&B training logs.

Align each metric at or before the slower run’s latest training step.
The positional inputs are the method run and baseline run, respectively.
"""
import sys, json, argparse
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

STEP_KEY = "trainer/global_step"


def parse_series(path):
    """Return {key: [(step, value), ...]} sorted by step, parsed from a .wandb file."""
    ds = DataStore()
    ds.open_for_scan(path)
    series = {}
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if rec.WhichOneof("record_type") != "history":
            continue
        d = {}
        for it in rec.history.item:
            key = it.key if it.key else "/".join(it.nested_key)
            try:
                d[key] = json.loads(it.value_json)
            except Exception:
                continue
        step = d.get(STEP_KEY)
        if step is None:
            continue
        for k, v in d.items():
            if isinstance(v, (int, float)):
                series.setdefault(k, []).append((step, v))
    for k in series:
        series[k].sort(key=lambda t: t[0])
    return series


def latest_step(series):
    s = series.get("train/psnr") or series.get(STEP_KEY)
    return s[-1][0] if s else None


def value_at(series, key, S):
    """Latest value for key at step <= S, plus the step it came from."""
    pts = series.get(key)
    if not pts:
        return None, None
    best = None
    for step, v in pts:
        if step <= S:
            best = (step, v)
        else:
            break
    return best if best else (None, None)


def fmt(x):
    if x is None:
        return "   —    "
    if isinstance(x, float):
        if abs(x) != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
            return f"{x:.3e}"
        return f"{x:.5g}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method")
    ap.add_argument("baseline")
    ap.add_argument("--method-label", "--core-label", default="PaveSplat")
    ap.add_argument("--base-label", default="SurfSplat")
    a = ap.parse_args()

    method = parse_series(a.method)
    base = parse_series(a.baseline)
    c_latest, b_latest = latest_step(method), latest_step(base)
    if c_latest is None or b_latest is None:
        print("ERROR: one run has no history yet "
              f"(method_latest={c_latest}, base_latest={b_latest})")
        return 1
    slower = a.base_label if b_latest <= c_latest else a.method_label
    S = min(c_latest, b_latest)

    print(f"# RUN COMPARISON @ slower-run latest step")
    print(f"#   {a.method_label}: latest train step {c_latest}")
    print(f"#   {a.base_label}: latest train step {b_latest}")
    print(f"#   slower run = {slower}  ->  align both at step <= {S}")
    print()

    # union of all numeric keys, drop noise
    keys = sorted(set(method) | set(base))
    drop_prefix = ("_", "info/", "lr-AdamW/")
    keys = [k for k in keys if not k.startswith(drop_prefix) and k != STEP_KEY]
    # group order: train, loss, diag, others
    order = {"train/": 0, "loss/": 1, "diag/": 2}
    keys.sort(key=lambda k: (order.get(k.split("/")[0] + "/", 3), k))

    w = max(len(k) for k in keys)
    print(f"{'metric'.ljust(w)} | {a.base_label:>11} | {a.method_label:>11} | {'Δ(method-base)':>12} | step(b/c)")
    print("-" * (w + 52))
    for k in keys:
        bs, bv = value_at(base, k, S)
        cs, cv = value_at(method, k, S)
        delta = ""
        if isinstance(bv, (int, float)) and isinstance(cv, (int, float)):
            delta = fmt(cv - bv)
        steps = f"{bs if bs is not None else '-'}/{cs if cs is not None else '-'}"
        print(f"{k.ljust(w)} | {fmt(bv):>11} | {fmt(cv):>11} | {delta:>12} | {steps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
