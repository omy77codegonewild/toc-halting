"""Bounded-halting prediction experiments.

    python experiment.py            # full run (a few minutes, depending on CPU)
    python experiment.py --quick    # smaller smoke-test run (~2 min); overwrites results/

Label for a step limit L:  1 = "halts within L steps", 0 = "has not halted within L steps".
(0 deliberately does NOT mean "never halts".)

Two populations are studied (generators live in tm.sample_machine):
  random : small random machines (2-6 states) + random binary inputs.
           Almost everything that halts does so within a few steps.
  hard   : bigger machines (4-10 states, fewer halting rules), keeping ONLY pairs that are
           still unresolved after the cheap probe (PROBE_STEPS steps). This is where a
           learned predictor adds real value as a triage tool.

Every study is repeated over 5 stratified train/test splits (SEEDS) and reported as mean +- std.

Outputs (in results/):
  dataset_random.csv, dataset_hard.csv   machines (rules), inputs, probe traces, features, truth
  results.csv / results_summary.csv      per-seed / aggregated metrics for every L, model, feature set
  complexity.csv                         generalisation to unseen machine sizes
  feature_importance.csv                 Random Forest importances (mean +- std)
  probe_sweep.csv                        effect of the probe budget k
  mlp_ablation.csv                       how the MLP's class-imbalance fix matters
  ood_benchmarks.csv                     hand-written / busy-beaver machines
  cost.csv                               ms per pair: simulating L steps vs features + prediction
  models.joblib                          models trained on all data (used by app.py)
  summary.md                             key tables in Markdown (source for REPORT.md)
  *.png                                  plots
"""
from __future__ import annotations

import argparse
import os
import random
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from features import PROBE_COLS, PROBE_STEPS, STATIC_COLS, all_features, static_features
from ml_models import MLP_VARIANTS, BalancedMLP, make_models
from tm import (
    HARD_P_HALT,
    HARD_STATES,
    PRESETS,
    RANDOM_P_HALT,
    RANDOM_STATES,
    format_rules_compact,
    parse_machine,
    random_input,
    run,
    sample_machine,
)

OUT = Path(__file__).parent / "results"
LIMITS = [10, 25, 50, 100, 250, 500, 1000]
MAX_LIMIT = max(LIMITS)
REPORT_L = 250  # the step limit used for single-L tables and studies
SEEDS = [42, 43, 44, 45, 46]
BASE_SEED = 42
warnings.filterwarnings("ignore")

PROBE_FS = f"static + {PROBE_STEPS}-step probe"
FEATURE_SETS = {
    "static only": STATIC_COLS,
    PROBE_FS: STATIC_COLS + PROBE_COLS,
}
FULL_COLS = STATIC_COLS + PROBE_COLS
MODELS = ["Decision Tree", "Random Forest", "Neural Net (MLP)"]


# ------------------------------------------------------------------ dataset
def _format_trace_preview(trace: list) -> str:
    return " -> ".join(f"{step}:{state}@{head}" for step, state, head, _ in trace)


def build_dataset(n: int, population: str, seed: int = BASE_SEED) -> tuple[pd.DataFrame, list]:
    """Sample n unique (machine, input) pairs and record ground truth up to MAX_LIMIT steps.

    Returns the dataframe and the list of (machine, input) pairs (reused by the probe sweep).
    """
    hard = population == "hard"
    rng = random.Random(seed + (1000 if hard else 0))
    rows, pairs = [], []
    seen_pairs: set[tuple[str, str]] = set()
    while len(rows) < n:
        tm = sample_machine(rng, population)
        inp = random_input(rng)
        rules_str = format_rules_compact(tm)
        if (rules_str, inp) in seen_pairs:
            continue  # keep pairs unique so train and test never share an example

        probe_res = run(tm, inp, PROBE_STEPS, record=True)
        if hard and probe_res.status != "limit":
            continue  # already resolved by the cheap probe -> not interesting
        seen_pairs.add((rules_str, inp))

        res = run(tm, inp, MAX_LIMIT)  # ground truth by simulation
        row = {
            "rules": rules_str,
            "input_str": inp,
            "probe_trace": _format_trace_preview(probe_res.trace),
        }
        row.update(all_features(tm, inp))
        row["status"] = res.status
        row["halt_step"] = res.steps if res.status == "halted" else -1
        rows.append(row)
        pairs.append((tm, inp))
    return pd.DataFrame(rows), pairs


def label(df: pd.DataFrame, limit: int) -> np.ndarray:
    """1 = halted within `limit` steps, 0 = has not halted within `limit` steps."""
    return ((df["halt_step"] >= 0) & (df["halt_step"] <= limit)).astype(int).values


def split(n: int, strat: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified 70/30 train/test split of row indices."""
    return train_test_split(np.arange(n), test_size=0.3, random_state=seed, stratify=strat)


def _quietly(fn, seed: int) -> list[dict]:
    # Worker processes don't inherit warning filters. The MLP's "max iterations reached"
    # warning is expected: we deliberately train for a fixed 60 epochs (see ml_models.py).
    warnings.filterwarnings("ignore")
    return fn(seed)


def over_seeds(one_seed) -> list[dict]:
    """Run `one_seed(seed) -> list of result rows` for every seed and concatenate the rows.

    The seeds are independent (each one is its own train/test split), so they run in
    parallel worker processes. Every seed is computed exactly as it would be sequentially,
    so the results are identical; only the wall-clock time shrinks.
    """
    n_jobs = min(len(SEEDS), os.cpu_count() or 1)
    per_seed = Parallel(n_jobs=n_jobs)(delayed(_quietly)(one_seed, seed) for seed in SEEDS)
    return [row for rows in per_seed for row in rows]


# ------------------------------------------------------------------ metrics
def scores(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    has_both = len(np.unique(y_true)) == 2
    if y_prob is not None and has_both:
        roc = float(roc_auc_score(y_true, y_prob))
        pr = float(average_precision_score(y_true, y_prob))
    else:
        roc = pr = np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc,
        "pr_auc": pr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def prob_pos(model, X: np.ndarray) -> np.ndarray:
    """Predicted probability of class 1 ('halts within L')."""
    probs = model.predict_proba(X)
    if probs.shape[1] == 2:
        return probs[:, 1]
    return np.full(len(X), float(model.classes_[0]))  # model only ever saw one class


def fit_and_score(model, Xtr, ytr, Xte, yte) -> dict:
    t1 = time.time()
    model.fit(Xtr, ytr)
    train_s = time.time() - t1
    t2 = time.time()
    pred = model.predict(Xte)
    prob = prob_pos(model, Xte)
    infer_ms = (time.time() - t2) / len(Xte) * 1000
    return dict(**scores(yte, pred, prob), train_s=train_s, infer_ms=infer_ms)


# ------------------------------------------------------------------ studies
def run_study(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Train/evaluate every model at every step limit, over all split seeds."""
    strat = (df["halt_step"] >= 0).astype(int)

    def one_seed(seed: int) -> list[dict]:
        rows = []
        idx_train, idx_test = split(len(df), strat, seed)
        # If two step limits give exactly the same labels (e.g. nothing new halts between
        # L=250 and L=500), every model would be trained identically -> reuse the results.
        cache: dict[bytes, list[dict]] = {}
        for L in LIMITS:
            y = label(df, L)
            if y.tobytes() in cache:
                rows += [dict(r, limit=L) for r in cache[y.tobytes()]]
                continue
            ytr, yte = y[idx_train], y[idx_test]
            base = dict(population=name, seed=seed, limit=L, n_pos_test=int(yte.sum()), n_test=len(yte))
            block = []

            # Baseline 1: always predict the most common training label
            maj = np.full_like(yte, int(ytr.mean() >= 0.5))
            block.append(dict(base, features="-", model="Majority baseline",
                              **scores(yte, maj, maj.astype(float)), train_s=0.0, infer_ms=0.0))
            # Baseline 2: "halts" iff the cheap probe run already halted
            rule = df["probe_halted"].values[idx_test]
            block.append(dict(base, features="-", model=f"Rule: simulate {PROBE_STEPS} steps",
                              **scores(yte, rule, rule.astype(float)), train_s=0.0, infer_ms=0.0))

            if ytr.sum() >= 5:  # need some positives to learn anything
                for fs_name, cols in FEATURE_SETS.items():
                    Xtr, Xte = df[cols].values[idx_train], df[cols].values[idx_test]
                    for m_name, model in make_models(seed).items():
                        block.append(dict(base, features=fs_name, model=m_name,
                                          **fit_and_score(model, Xtr, ytr, Xte, yte)))
            cache[y.tobytes()] = block
            rows += block
        return rows

    return pd.DataFrame(over_seeds(one_seed))


def summarize(res: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """Mean and std over seeds -> columns '<metric>_mean', '<metric>_std'."""
    agg = res.groupby(keys, sort=False)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns.to_flat_index()]
    return agg.reset_index()


def summarize_results(res: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "train_s", "infer_ms"]
    out = summarize(res, ["population", "limit", "features", "model"], metrics)
    counts = res.groupby(["population", "limit", "features", "model"], sort=False)[["n_pos_test", "n_test"]].mean()
    return out.merge(counts.reset_index(), on=["population", "limit", "features", "model"])


def complexity_study(dfs: dict[str, pd.DataFrame], L: int = REPORT_L) -> pd.DataFrame:
    """Train on small machines only, then test on every machine size (incl. unseen bigger ones)."""
    rows = []
    max_train_states = {"random": 4, "hard": 6}
    for pop_name, df in dfs.items():
        y = label(df, L)
        ns_col = df["n_states"].values
        cutoff = max_train_states[pop_name]

        def one_seed(seed: int) -> list[dict]:
            out = []
            idx_train, _ = split(len(df), (df["halt_step"] >= 0), seed)
            is_train = np.isin(np.arange(len(df)), idx_train)
            tr = is_train & (ns_col <= cutoff)  # train only on SMALL machines
            for fs_name, cols in FEATURE_SETS.items():
                X = df[cols].values
                for m_name, model in make_models(seed).items():
                    model.fit(X[tr], y[tr])
                    for ns in sorted(np.unique(ns_col)):
                        # seen sizes: only test rows; unseen sizes: every row (none were trained on)
                        te = (ns_col == ns) & (~is_train if ns <= cutoff else True)
                        out.append(dict(
                            population=pop_name, seed=seed, train_max_states=cutoff,
                            features=fs_name, model=m_name, n_states=int(ns), seen=bool(ns <= cutoff),
                            n_test=int(te.sum()), n_pos=int(y[te].sum()),
                            **scores(y[te], model.predict(X[te]), prob_pos(model, X[te])),
                        ))
            return out

        rows += over_seeds(one_seed)
    return pd.DataFrame(rows)


def feature_importance_study(dfs: dict[str, pd.DataFrame], L: int = REPORT_L) -> pd.DataFrame:
    """Random Forest (Gini) importances, averaged over the training splits of all seeds."""
    rows = []
    for pop_name, df in dfs.items():
        y = label(df, L)
        for seed in SEEDS:
            idx_train, _ = split(len(df), (df["halt_step"] >= 0), seed)
            rf = make_models(seed)["Random Forest"]
            rf.fit(df[FULL_COLS].values[idx_train], y[idx_train])
            for feat, imp in zip(FULL_COLS, rf.feature_importances_):
                rows.append(dict(population=pop_name, seed=seed, feature=feat,
                                 group="static" if feat in STATIC_COLS else "probe", importance=float(imp)))
    fi = pd.DataFrame(rows)
    agg = fi.groupby(["population", "feature", "group"])["importance"].agg(["mean", "std"]).reset_index()
    agg = agg.rename(columns={"mean": "importance", "std": "importance_std"})
    return agg.sort_values(["population", "importance"], ascending=[True, False]).reset_index(drop=True)


def probe_sweep_study(df_hard: pd.DataFrame, pairs: list, L: int = REPORT_L) -> pd.DataFrame:
    """How much does a longer probe run (k steps) help on the hard population?"""
    y = label(df_hard, L)
    rows = []
    for k in [0, 5, 10, 25, 50]:
        # k = 0 means static features only (no execution at all)
        feats = [static_features(tm, inp) if k == 0 else all_features(tm, inp, k=k) for tm, inp in pairs]
        X = pd.DataFrame(feats).values

        def one_seed(seed: int) -> list[dict]:
            idx_tr, idx_te = split(len(y), y, seed)
            return [dict(probe_k=k, limit=L, seed=seed, model=m_name,
                         **fit_and_score(model, X[idx_tr], y[idx_tr], X[idx_te], y[idx_te]))
                    for m_name, model in make_models(seed).items()]

        rows += over_seeds(one_seed)
    return pd.DataFrame(rows)


def mlp_ablation_study(dfs: dict[str, pd.DataFrame], L: int = REPORT_L) -> pd.DataFrame:
    """Compare ways of making the MLP cope with class imbalance (static + probe features)."""
    rows = []
    for pop_name, df in dfs.items():
        y = label(df, L)
        X = df[FULL_COLS].values

        def one_seed(seed: int) -> list[dict]:
            out = []
            idx_tr, idx_te = split(len(df), (df["halt_step"] >= 0), seed)
            for variant, opts in MLP_VARIANTS.items():
                model = BalancedMLP(random_state=seed, **opts)
                r = fit_and_score(model, X[idx_tr], y[idx_tr], X[idx_te], y[idx_te])
                out.append(dict(population=pop_name, seed=seed, limit=L, variant=variant,
                                threshold=model.threshold_, **r))
            return out

        rows += over_seeds(one_seed)
    return pd.DataFrame(rows)


def cost_study(pairs: list, app_models: dict, n: int = 1000) -> pd.DataFrame:
    """Wall-clock cost per (machine, input): simulating L steps vs. features + prediction.

    Simulation cost grows with L (unresolved machines run the full budget); the ML
    triage cost is fixed (static features + a PROBE_STEPS-step probe + one prediction).
    """
    sample = pairs[:n]
    rows = []
    for L in LIMITS:
        t = time.perf_counter()
        for tm, inp in sample:
            run(tm, inp, L)
        rows.append(dict(method=f"simulate L={L} steps", limit=L, ms_per_pair=(time.perf_counter() - t) / len(sample) * 1000))
    t = time.perf_counter()
    X = np.array([[f[c] for c in FULL_COLS] for f in (all_features(tm, inp) for tm, inp in sample)])
    feat_ms = (time.perf_counter() - t) / len(sample) * 1000
    rows.append(dict(method=f"features (static + {PROBE_STEPS}-step probe)", limit=np.nan, ms_per_pair=feat_ms))
    for m_name in MODELS:
        clf = app_models["hard"][m_name][REPORT_L]
        t = time.perf_counter()
        clf.predict_proba(X)  # batched, as a triage pipeline would do it
        rows.append(dict(method=f"predict ({m_name}, batched)", limit=np.nan,
                         ms_per_pair=(time.perf_counter() - t) / len(sample) * 1000))
    return pd.DataFrame(rows)


def train_app_models(dfs: dict[str, pd.DataFrame]) -> dict:
    """Train every model on ALL data of each population at each L (for the Streamlit app)."""
    app_models: dict = {}
    for pop_name, df in dfs.items():
        app_models[pop_name] = {m: {} for m in MODELS}
        X = df[FULL_COLS].values
        cache: dict[bytes, dict] = {}  # identical labels -> identical models, train once
        for L in LIMITS:
            y = label(df, L)
            if y.tobytes() not in cache:
                if y.sum() < 5:
                    # e.g. hard population at L=10: by construction nothing halts that early
                    trained = {m: DummyClassifier(strategy="constant", constant=0).fit(X, y) for m in MODELS}
                else:
                    trained = {m: clf.fit(X, y) for m, clf in make_models(BASE_SEED).items()}
                cache[y.tobytes()] = trained
            for m in MODELS:
                app_models[pop_name][m][L] = cache[y.tobytes()][m]
    return app_models


def ood_benchmark_study(app_models: dict) -> pd.DataFrame:
    """Evaluate the app models on hand-written machines (never seen in training)."""
    rows = []
    for name, (rules_txt, inp) in PRESETS.items():
        tm, err = parse_machine(rules_txt)
        assert tm is not None and not err
        feats = all_features(tm, inp)
        X = np.array([[feats[c] for c in FULL_COLS]])
        for L in LIMITS:
            res = run(tm, inp, L, exact_loop_check=True)
            actual = int(res.status == "halted")
            for pop_name in app_models:
                for m_name in MODELS:
                    clf = app_models[pop_name][m_name][L]
                    pred = int(clf.predict(X)[0])  # uses the model's own decision threshold
                    rows.append(dict(
                        preset=name, limit=L, actual_status=res.status, actual_steps=res.steps,
                        actual_halts=actual, train_pop=pop_name, model=m_name,
                        prob_halt=float(prob_pos(clf, X)[0]), pred_halts=pred, correct=int(pred == actual),
                    ))
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- plots
def _style(ax, xlabel, ylabel, title, logx=False):
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


def make_plots(summary, cx, fi, sweep, ablation, ood) -> None:
    n_seeds = len(SEEDS)

    # 1. Accuracy vs step limit, both populations, with +-1 std bands
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row, (pop, metric) in enumerate([("random", "accuracy"), ("hard", "f1")]):
        sub = summary[summary["population"] == pop]
        for ax, fs in zip(axes[row], FEATURE_SETS):
            for m in MODELS:
                d = sub[(sub["features"] == fs) & (sub["model"] == m)].sort_values("limit")
                std = d[f"{metric}_std"].fillna(0)
                ax.plot(d["limit"], d[f"{metric}_mean"], marker="o", label=m)
                ax.fill_between(d["limit"], d[f"{metric}_mean"] - std, d[f"{metric}_mean"] + std, alpha=0.15)
            for m, st in [("Majority baseline", ":"), (f"Rule: simulate {PROBE_STEPS} steps", "--")]:
                d = sub[sub["model"] == m].sort_values("limit")
                ax.plot(d["limit"], d[f"{metric}_mean"], st, color="gray", label=m)
            _style(ax, "step limit L", f"{metric} (mean ± std, {n_seeds} seeds)", f"{pop}: features = {fs}", logx=True)
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Predicting 'halts within L steps' vs the step limit L")
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_limit.png", dpi=130)
    plt.close(fig)

    # 2. Hard cases: F1, PR-AUC (error bars) + number of positives per L
    hard = summary[summary["population"] == "hard"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for m in MODELS:
        d = hard[(hard["features"] == PROBE_FS) & (hard["model"] == m)].sort_values("limit")
        axes[0].errorbar(d["limit"], d["f1_mean"], yerr=d["f1_std"].fillna(0), marker="o", capsize=3, label=m)
        axes[1].errorbar(d["limit"], d["pr_auc_mean"], yerr=d["pr_auc_std"].fillna(0), marker="s", capsize=3, label=m)
    d_maj = hard[hard["model"] == "Majority baseline"].sort_values("limit")
    axes[0].plot(d_maj["limit"], d_maj["f1_mean"], ":", color="gray", label="baselines (F1 = 0)")
    axes[1].plot(d_maj["limit"], d_maj["pr_auc_mean"], ":", color="gray", label="baselines (= positive rate)")
    _style(axes[0], "step limit L", "F1", f"Hard cases: F1 ({PROBE_FS})", logx=True)
    _style(axes[1], "step limit L", "PR-AUC", f"Hard cases: PR-AUC ({PROBE_FS})", logx=True)
    for ax in axes[:2]:
        ax.legend(fontsize=8)
    axes[2].bar(range(len(d_maj)), d_maj["n_pos_test"], tick_label=d_maj["limit"].astype(str))
    _style(axes[2], "step limit L", "# positives in test set", "Slow halters resolved by step L")
    fig.tight_layout()
    fig.savefig(OUT / "hard_cases.png", dpi=130)
    plt.close(fig)

    # 3. Complexity generalisation (mean +- std over seeds)
    cxs = summarize(cx, ["population", "features", "model", "n_states"], ["accuracy", "f1", "roc_auc"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, pop, metric in [(axes[0], "random", "accuracy"), (axes[1], "hard", "f1")]:
        sub = cxs[cxs["population"] == pop]
        for m in MODELS:
            d = sub[(sub["features"] == PROBE_FS) & (sub["model"] == m)].sort_values("n_states")
            ax.errorbar(d["n_states"], d[f"{metric}_mean"], yerr=d[f"{metric}_std"].fillna(0),
                        marker="o", capsize=3, label=f"{m} (probe)")
        d = sub[(sub["features"] == "static only") & (sub["model"] == "Random Forest")].sort_values("n_states")
        ax.errorbar(d["n_states"], d[f"{metric}_mean"], yerr=d[f"{metric}_std"].fillna(0),
                    ls="--", marker="^", color="gray", capsize=3, label="Random Forest (static only)")
        cutoff = cx[cx["population"] == pop]["train_max_states"].iloc[0]
        ax.axvline(cutoff + 0.5, color="black", ls=":", lw=1, label="unseen sizes ->")
        _style(ax, "number of states", metric, f"{pop}: trained on <= {cutoff} states (L={REPORT_L})")
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_complexity.png", dpi=130)
    plt.close(fig)

    # 4. Feature importance (top 12 per population, with std error bars)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, pop in zip(axes, ["random", "hard"]):
        d = fi[fi["population"] == pop].head(12).iloc[::-1]
        colors = ["#1f77b4" if g == "static" else "#ff7f0e" for g in d["group"]]
        ax.barh(d["feature"], d["importance"], xerr=d["importance_std"], color=colors, capsize=2)
        _style(ax, "Gini importance (mean ± std)", "", f"Top 12 Random Forest features ({pop}, L={REPORT_L})")
    axes[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#1f77b4"), plt.Rectangle((0, 0), 1, 1, color="#ff7f0e")],
                   labels=["static", "probe"], fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=130)
    plt.close(fig)

    # 5. Probe budget sweep
    sw = summarize(sweep, ["probe_k", "model"], ["f1", "pr_auc"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for m in MODELS:
        d = sw[sw["model"] == m].sort_values("probe_k")
        axes[0].errorbar(d["probe_k"], d["f1_mean"], yerr=d["f1_std"], marker="o", capsize=3, label=m)
        axes[1].errorbar(d["probe_k"], d["pr_auc_mean"], yerr=d["pr_auc_std"], marker="s", capsize=3, label=m)
    _style(axes[0], "probe budget k (0 = static only)", "F1", f"Hard cases: F1 vs probe steps (L={REPORT_L})")
    _style(axes[1], "probe budget k (0 = static only)", "PR-AUC", f"Hard cases: PR-AUC vs probe steps (L={REPORT_L})")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "probe_sweep.png", dpi=130)
    plt.close(fig)

    # 6. MLP imbalance ablation (grouped bars, hard population)
    ab = summarize(ablation, ["population", "variant"], ["precision", "recall", "f1"])
    ab = ab[ab["population"] == "hard"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.27
    for i, metric in enumerate(["precision", "recall", "f1"]):
        ax.bar(np.arange(len(ab)) + (i - 1) * width, ab[f"{metric}_mean"], width,
               yerr=ab[f"{metric}_std"], capsize=3, label=metric)
    ax.set_xticks(np.arange(len(ab)), ab["variant"], fontsize=8)
    _style(ax, "", "score (mean ± std)", f"MLP class-imbalance fixes (hard, L={REPORT_L})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "mlp_ablation.png", dpi=130)
    plt.close(fig)

    # 7. Out-of-distribution heatmap: P(halt within L) per preset and model
    o = ood[ood["limit"] == REPORT_L].copy()
    o["column"] = o["train_pop"] + " / " + o["model"].str.replace("Neural Net (MLP)", "MLP", regex=False)
    grid = o.pivot_table(index="preset", columns="column", values="prob_halt", sort=False)
    truth = o.drop_duplicates("preset").set_index("preset")["actual_status"]
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(grid) + 2))
    im = ax.imshow(grid.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(grid.shape[1]), grid.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(grid.shape[0]), [f"{p}  [{truth[p]}]" for p in grid.index], fontsize=8)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label=f"predicted P(halts within {REPORT_L})")
    ax.set_title(f"Hand-written machines (out of distribution), L = {REPORT_L}; [true outcome]", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ood_heatmap.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------ markdown summary
def _pm(mean, std, digits=3) -> str:
    if pd.isna(mean):
        return "n/a"
    return f"{mean:.{digits}f} ± {0 if pd.isna(std) else std:.{digits}f}"


def write_summary_md(dfs, summary, cx, fi, sweep, ablation, ood, cost, mlp_thresholds, runtime_s) -> None:
    """Key numbers in Markdown so the report can quote real results verbatim."""
    L = REPORT_L
    md = [f"# Experiment summary (auto-generated by experiment.py)\n",
          f"Seeds: {SEEDS} (stratified 70/30 splits). Runtime: {runtime_s:.0f} s.\n"]

    md.append("## Datasets\n\n| population | pairs | halted | looping | limit (1000) | halts within 250 |\n|---|---|---|---|---|---|")
    for pop, df in dfs.items():
        vc = df["status"].value_counts()
        md.append(f"| {pop} | {len(df)} | {vc.get('halted', 0)} | {vc.get('looping', 0)} | "
                  f"{vc.get('limit', 0)} | {label(df, 250).sum()} ({label(df, 250).mean():.1%}) |")

    md.append("\n## Test positives per step limit (mean over seeds)\n\n| population | " +
              " | ".join(f"L={l}" for l in LIMITS) + " |\n|---|" + "---|" * len(LIMITS))
    for pop in dfs:
        d = summary[(summary["population"] == pop) & (summary["model"] == "Majority baseline")].sort_values("limit")
        md.append(f"| {pop} | " + " | ".join(f"{v:.0f} / {t:.0f}" for v, t in zip(d["n_pos_test"], d["n_test"])) + " |")

    for pop in dfs:
        d = summary[(summary["population"] == pop) & (summary["limit"] == L)]
        md.append(f"\n## {pop} population, L = {L}\n\n| features | model | accuracy | precision | recall | F1 | ROC-AUC | PR-AUC |\n|---|---|---|---|---|---|---|---|")
        for _, r in d.iterrows():
            md.append(f"| {r['features']} | {r['model']} | " + " | ".join(
                _pm(r[f"{m}_mean"], r[f"{m}_std"]) for m in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]) + " |")

    md.append(f"\n## Metric vs step limit ({PROBE_FS})\n")
    for pop, metric in [("random", "accuracy"), ("hard", "f1")]:
        md.append(f"\n**{pop}: {metric}**\n\n| model | " + " | ".join(f"L={l}" for l in LIMITS) + " |\n|---|" + "---|" * len(LIMITS))
        sub = summary[summary["population"] == pop]
        for m in MODELS + [f"Rule: simulate {PROBE_STEPS} steps"]:
            d = sub[(sub["model"] == m) & (sub["features"].isin([PROBE_FS, "-"]))].set_index("limit")
            # a limit can be missing (no positives to train on) -> show "n/a"
            md.append(f"| {m} | " + " | ".join(
                _pm(d.at[l, f"{metric}_mean"], d.at[l, f"{metric}_std"]) if l in d.index else "n/a"
                for l in LIMITS) + " |")

    md.append(f"\n## Complexity generalisation (L = {L}, mean over seeds)\n\n| population | features | model | n_states | seen in training | accuracy | recall | F1 | ROC-AUC |\n|---|---|---|---|---|---|---|---|---|")
    cxs = summarize(cx, ["population", "features", "model", "n_states", "seen"], ["accuracy", "recall", "f1", "roc_auc"])
    for _, r in cxs.iterrows():
        md.append(f"| {r['population']} | {r['features']} | {r['model']} | {r['n_states']} | {'yes' if r['seen'] else 'no'} | "
                  + " | ".join(_pm(r[f"{m}_mean"], r[f"{m}_std"]) for m in ["accuracy", "recall", "f1", "roc_auc"]) + " |")

    md.append(f"\n## Random Forest feature importance (L = {L}, top 8)\n")
    for pop in dfs:
        d = fi[fi["population"] == pop].head(8)
        md.append(f"\n**{pop}**: " + ", ".join(f"`{r.feature}` {r.importance:.3f}" for r in d.itertuples()))

    md.append(f"\n\n## Probe budget sweep (hard, L = {L})\n\n| k | model | F1 | PR-AUC |\n|---|---|---|---|")
    sw = summarize(sweep, ["probe_k", "model"], ["f1", "pr_auc"])
    for _, r in sw.iterrows():
        md.append(f"| {r['probe_k']} | {r['model']} | {_pm(r['f1_mean'], r['f1_std'])} | {_pm(r['pr_auc_mean'], r['pr_auc_std'])} |")

    md.append(f"\n## MLP imbalance ablation (L = {L}, static + probe)\n\n| population | variant | precision | recall | F1 | PR-AUC | train s |\n|---|---|---|---|---|---|---|")
    ab = summarize(ablation, ["population", "variant"], ["precision", "recall", "f1", "pr_auc", "train_s"])
    for _, r in ab.iterrows():
        md.append(f"| {r['population']} | {r['variant']} | " + " | ".join(
            _pm(r[f"{m}_mean"], r[f"{m}_std"]) for m in ["precision", "recall", "f1", "pr_auc"]) + f" | {r['train_s_mean']:.2f} |")

    md.append(f"\n## Out-of-distribution hand-written machines (L = {L})\n\n| machine | true outcome (steps) | " +
              " | ".join(f"{p} {m}" for p in dfs for m in ["DT", "RF", "MLP"]) + " |\n|---|---|" + "---|" * 6)
    o = ood[ood["limit"] == L]
    for preset, g in o.groupby("preset", sort=False):
        cells = []
        for pop in dfs:
            for m in MODELS:
                r = g[(g["train_pop"] == pop) & (g["model"] == m)].iloc[0]
                cells.append(f"{r['prob_halt']:.2f}{'' if r['correct'] else ' ✗'}")
        first = g.iloc[0]
        md.append(f"| {preset} | {first['actual_status']} ({first['actual_steps']}) | " + " | ".join(cells) + " |")
    md.append("\nCells: predicted P(halts within L); ✗ = the model's yes/no prediction is wrong.")
    md.append("\n**Decision thresholds of the MLP app models:** " + ", ".join(
        f"{pop} L={lim}: {thr:.3f}" for (pop, lim), thr in mlp_thresholds.items()))
    acc = ood.groupby(["train_pop", "model"])["correct"].mean()
    md.append("\n**OOD accuracy over all presets and all L:** " +
              ", ".join(f"{p}/{m}: {v:.0%}" for (p, m), v in acc.items()))

    md.append("\n## Cost per (machine, input) pair: simulation vs ML triage (hard pairs, ms)\n\n| method | ms per pair |\n|---|---|")
    for _, r in cost.iterrows():
        md.append(f"| {r['method']} | {r['ms_per_pair']:.4f} |")

    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# --------------------------------------------------------------------- main
def main(n_random: int, n_hard: int) -> None:
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    log(f"datasets: random={n_random} pairs ({RANDOM_STATES[0]}-{RANDOM_STATES[1]} states, p_halt {RANDOM_P_HALT}), "
        f"hard={n_hard} pairs ({HARD_STATES[0]}-{HARD_STATES[1]} states, p_halt {HARD_P_HALT}, unresolved after {PROBE_STEPS} steps)")
    df_rnd, _ = build_dataset(n_random, "random")
    df_hard, hard_pairs = build_dataset(n_hard, "hard")
    dfs = {"random": df_rnd, "hard": df_hard}
    for pop, df in dfs.items():
        df.to_csv(OUT / f"dataset_{pop}.csv", index=False)
        log(f"  {pop}: {df['status'].value_counts().to_dict()}")

    log(f"main study: {len(SEEDS)} seeds x {len(LIMITS)} step limits x 2 feature sets x 3 models")
    res = pd.concat([run_study(df, pop) for pop, df in dfs.items()], ignore_index=True)
    res.to_csv(OUT / "results.csv", index=False)
    summary = summarize_results(res)
    summary.to_csv(OUT / "results_summary.csv", index=False)

    log("complexity generalisation study")
    cx = complexity_study(dfs)
    cx.to_csv(OUT / "complexity.csv", index=False)

    log("feature importance")
    fi = feature_importance_study(dfs)
    fi.to_csv(OUT / "feature_importance.csv", index=False)

    log("probe budget sweep (hard population)")
    sweep = probe_sweep_study(df_hard, hard_pairs)
    sweep.to_csv(OUT / "probe_sweep.csv", index=False)

    log("MLP class-imbalance ablation")
    ablation = mlp_ablation_study(dfs)
    ablation.to_csv(OUT / "mlp_ablation.csv", index=False)

    log("training app models on all data")
    app_models = train_app_models(dfs)
    joblib.dump({"models": app_models, "cols": FULL_COLS, "feature_importance": fi},
                OUT / "models.joblib", compress=3)

    log("cost of simulation vs ML triage")
    cost = cost_study(hard_pairs, app_models)
    cost.to_csv(OUT / "cost.csv", index=False)

    log("out-of-distribution benchmarks, plots, summary.md")
    ood = ood_benchmark_study(app_models)
    ood.to_csv(OUT / "ood_benchmarks.csv", index=False)
    make_plots(summary, cx, fi, sweep, ablation, ood)
    # the MLP decides with a tuned threshold (not 0.5); record it for the report
    mlp_thresholds = {(pop, lim): getattr(app_models[pop]["Neural Net (MLP)"][lim], "threshold_", 0.5)
                      for pop in dfs for lim in (25, 100, 250)}
    write_summary_md(dfs, summary, cx, fi, sweep, ablation, ood, cost, mlp_thresholds, time.time() - t0)
    log(f"all done -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-random", type=int, default=6000, help="pairs in the random population")
    ap.add_argument("--n-hard", type=int, default=12000, help="pairs in the hard population")
    ap.add_argument("--quick", action="store_true", help="small run for smoke testing (1500 / 3000 pairs)")
    args = ap.parse_args()
    if args.quick:
        args.n_random, args.n_hard = 1500, 3000
    main(args.n_random, args.n_hard)
