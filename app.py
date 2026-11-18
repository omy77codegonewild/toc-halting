"""Turing Machine Lab: simulator + bounded halting prediction.

    streamlit run app.py
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from features import PROBE_COLS, PROBE_STEPS, STATIC_COLS, all_features
from tm import (
    HALT,
    PRESETS,
    format_machine,
    parse_machine,
    random_input,
    random_machine,
    run,
    sample_machine,
    to_dot,
)

RESULTS = Path(__file__).parent / "results"
LIMITS = [10, 25, 50, 100, 250, 500, 1000]
REPORT_L = 250
MODEL_NAMES = ["Decision Tree", "Random Forest", "Neural Net (MLP)"]
PROBE_FS = f"static + {PROBE_STEPS}-step probe"
RULE_MODEL = f"Rule: simulate {PROBE_STEPS} steps"
POP_LABEL = {
    "hard": f"hard: still running after {PROBE_STEPS} steps",
    "random": "random: any small machine",
}
MOVE_TEXT = {"L": "move left", "R": "move right", "S": "stay"}

FEATURE_HELP = {
    # static: read straight off the rule table and the input, nothing is executed
    "n_states": "number of states (not counting H)",
    "n_rules": "number of transition rules",
    "alphabet_size": "number of different tape symbols the rules use",
    "frac_to_halt": "share of rules that go to H",
    "frac_move_right": "share of rules that move the head right",
    "frac_move_stay": "share of rules that keep the head still (S)",
    "frac_write_blank": "share of rules that write a blank",
    "frac_self_loop": "share of rules that stay in the same state",
    "sweepers_right": "rules that keep state and symbol and move right (scan right)",
    "sweepers_left": "rules that keep state and symbol and move left (scan left)",
    "runaway_writers": "rules that write on a blank and move on in the same state (can run away forever)",
    "reachable_states": "states reachable from the start state in the state diagram",
    "dist_to_halt": "fewest arrows from the start state to H in the diagram (99 = H unreachable)",
    "halting_rules_reachable": "rules into H that start from a reachable state",
    "start_halting_rules": "rules of the start state that go straight to H",
    "input_len": "length of the input",
    "input_nonblank": "non-blank cells in the input",
    "input_unique_symbols": "different symbols in the input",
    "input_ones": "number of 1s in the input",
    # probe: behaviour during a short trial run
    "probe_halted": f"1 if it halted within the {PROBE_STEPS}-step trial run",
    "probe_looping": "1 if a configuration repeated during the trial run",
    "probe_steps": "steps executed in the trial run",
    "probe_distinct_states": "different states visited",
    "probe_head_span": "cells the head visited (rightmost - leftmost + 1)",
    "probe_net_move": "head position at the end minus at the start",
    "probe_nonblank_cells": "non-blank cells after the trial run",
    "probe_tape_growth": "change in non-blank cells compared with the input",
}

st.set_page_config(page_title="Turing Machine Lab", page_icon="🧮", layout="wide")


# ------------------------------------------------------------------ helpers
@st.cache_resource
def load_models():
    p = RESULTS / "models.joblib"
    return joblib.load(p) if p.exists() else None


@st.cache_data
def load_csv(name: str):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def prob_pos(model, X: np.ndarray) -> float:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.shape[1] == 2:
            return float(probs[0][1])
        return float(int(model.classes_[0]))
    return float(model.predict(X)[0])


def model_verdict(clf, X: np.ndarray) -> tuple[float, bool]:
    """(probability of 'halts within L', yes/no decision).

    The decision uses the model's own predict(): the MLP has a tuned threshold, not 0.5.
    """
    return prob_pos(clf, X), bool(clf.predict(X)[0])


def describe(res, limit: int) -> tuple[str, str]:
    """(kind, message) for a run result. kind in success|info|warning."""
    if res.status == "halted":
        return "success", f"**Halted** after {res.steps} steps."
    if res.status == "looping":
        return "info", (
            f"**Provably never halts.** The exact same configuration (state, head, tape) "
            f"occurred twice by step {res.steps}, so the machine is stuck in a cycle."
        )
    return "warning", (
        f"**Not halted within {limit} steps.** This is *not* the same as 'never halts' - "
        f"it might halt at step {limit + 1} or never. The halting problem is undecidable, "
        f"so no program can settle this in general."
    )


def tape_html(tape: dict, head: int, state: str, blank: str = "_", radius: int = 12) -> str:
    cells = []
    for i in range(head - radius, head + radius + 1):
        sym = tape.get(i, blank)
        style = (
            "display:inline-block;width:30px;height:34px;line-height:34px;text-align:center;"
            "font-family:monospace;font-size:18px;border:1px solid rgba(128,128,128,.5);"
            "margin-right:-1px;"
        )
        if i == head:
            style += "background:#f4b400;color:#000;border:2px solid #b37e00;font-weight:700;"
        cells.append(f'<span style="{style}">{sym}</span>')
    return (
        f'<div style="overflow-x:auto;white-space:nowrap;padding:6px 0">{"".join(cells)}</div>'
        f'<div style="font-family:monospace;margin-top:2px">head at cell {head} &nbsp;|&nbsp; '
        f"state <b>{state}</b> &nbsp;|&nbsp; blank <b>{blank}</b></div>"
    )


def explain_step(tm, res, trace: list, i: int) -> tuple[str, str]:
    """Plain-language description of what the machine does from snapshot i."""
    s_no, state, head, tape = trace[i]
    read = tape.get(head, tm.blank)
    at_end = i == len(trace) - 1 and s_no == res.steps
    if state == HALT:
        return "success", f"The machine is in the halt state **H**, so it stops. Total: **{s_no} steps**."
    rule = tm.transitions.get((state, read))
    if rule is None:
        return "success", (
            f"State **{state}** reads **{read}**, but there is **no rule** for ({state}, {read}). "
            f"A missing rule means the machine halts. Total: **{s_no} steps**."
        )
    w, m, n = rule
    action = (f"δ({state}, {read}) = ({w}, {m}, {n}): write **{w}**, **{MOVE_TEXT[m]}**, "
              f"go to state **{n}**.")
    if at_end and res.status == "looping":
        return "info", ("This exact configuration (state, head, tape) already appeared earlier, so "
                        f"the machine will repeat the same cycle forever. Next rule would be {action}")
    if at_end and res.status == "limit":
        return "warning", (f"Step budget used up after {s_no} steps. Next rule would be {action} "
                           "Whether it ever halts is unknown.")
    return "plain", f"State **{state}** reads **{read}** under the head, so it applies {action}"


def tape_symbols(tm) -> list[str]:
    syms = {a for _, a in tm.transitions} | {w for w, _, _ in tm.transitions.values()} | {tm.blank}
    return sorted(syms)


def delta_table(tm, active: tuple[str, str] | None):
    """Transition table δ (states x symbols) with the rule used at this step highlighted."""
    syms = tape_symbols(tm)
    states = tm.states
    data = {}
    for sym in syms:
        col = []
        for s in states:
            r = tm.transitions.get((s, sym))
            col.append("halts (no rule)" if r is None else f"{r[0]}, {r[1]}, {r[2]}")
        data[f"read {sym}"] = col
    df = pd.DataFrame(data, index=pd.Index(states, name="state"))

    def highlight(_):
        css = pd.DataFrame("", index=df.index, columns=df.columns)
        if active and active[0] in css.index and f"read {active[1]}" in css.columns:
            css.loc[active[0], f"read {active[1]}"] = "background-color:#f4b400;color:#000;font-weight:700"
        return css

    return df.style.apply(highlight, axis=None)


def trace_table(tm, trace: list) -> pd.DataFrame:
    rows = []
    for (s_no, state, head, tape), (_, _, _, tape_after) in zip(trace, trace[1:]):
        read = tape.get(head, tm.blank)
        w, m, n = tm.transitions[(state, read)]
        lo_hi = list(tape_after) or [0]
        tape_str = "".join(tape_after.get(c, tm.blank) for c in range(min(lo_hi), max(lo_hi) + 1))
        rows.append(dict(step=s_no + 1, state=state, head=head, read=read, write=w, move=m,
                         next_state=n, tape_after=tape_str))
    return pd.DataFrame(rows)


def summary_value(summ, pop, model, features, metric, limit=REPORT_L):
    if summ is None:
        return None
    d = summ[(summ["population"] == pop) & (summ["model"] == model)
             & (summ["features"] == features) & (summ["limit"] == limit)]
    return float(d[f"{metric}_mean"].iloc[0]) if len(d) else None


# ----------------------------------------------------------------- sidebar
st.sidebar.title("Machine")
st.sidebar.markdown("**1 · Pick a machine**")
choice = st.sidebar.selectbox("Choose a machine", list(PRESETS) + ["Random machine"])
n_states, seed = 4, 1
if choice == "Random machine":
    n_states = st.sidebar.slider("Number of states", 2, 10, 4)
    seed = st.sidebar.number_input("Random seed", 0, 10_000_000, 1)
    default_text = format_machine(random_machine(n_states, random.Random(seed)))
    default_inp = random_input(random.Random(seed + 7))
else:
    default_text, default_inp = PRESETS[choice]

st.sidebar.markdown("**2 · Edit it (optional)**")
c_cfg1, c_cfg2 = st.sidebar.columns(2)
start_state = c_cfg1.text_input(
    "Start state", value="q0", key=f"start::{choice}::{n_states}::{seed}",
    help="Leave as q0 or enter any state name present in your rules.",
)
blank_sym = c_cfg2.text_input(
    "Blank symbol", value="_", max_chars=1, key=f"blank::{choice}::{n_states}::{seed}",
    help="Single character representing an empty tape cell.",
)
rules_text = st.sidebar.text_area(
    "Transition rules",
    default_text,
    height=240,
    key=f"rules::{choice}::{n_states}::{seed}",
    help="One rule per line: `state read write move next`. Moves: L, R, S. "
    "Halt state is `H`. A missing rule also halts the machine.",
)
st.sidebar.caption(
    "One rule per line: `state read write move next`, e.g. `q0 1 0 R q1` means *in state q0 "
    "reading 1: write 0, move right, go to q1*. `H` is the halt state; a missing rule also "
    "halts; `#` starts a comment."
)
input_str = st.sidebar.text_input(
    "Input tape", default_inp, key=f"inp::{choice}::{n_states}::{seed}",
    help="Head starts on the first character. Supports any single-character symbols.",
)
st.sidebar.markdown("**3 · Step budget**")
max_steps = st.sidebar.slider(
    "Step limit for simulation", 10, 5000, 1000,
    help="The simulator stops after this many steps and reports 'not halted within L' if the "
         "machine is still running.",
)

tm, errors = parse_machine(
    rules_text,
    start=start_state.strip() or None,
    blank=(blank_sym or "_"),
)
st.title("🧮 Turing Machine Lab")
st.caption(
    "Simulate Turing machines step by step, then see how well machine learning can predict "
    "whether a machine halts **within a step limit L**. New here? Start with the **🏠 Overview** tab."
)
if errors:
    st.error("Could not parse the rules:\n\n" + "\n".join(f"- {e}" for e in errors))
    st.stop()

tab_home, tab_sim, tab_pred, tab_exp, tab_theory = st.tabs(
    ["🏠 Overview", "▶ Simulator", "🔮 Halting prediction", "📊 Experiments", "📚 Theory notes"]
)

# ---------------------------------------------------------------- overview
with tab_home:
    st.markdown(
        "This project has two parts. **A Turing machine simulator** runs any machine you define, "
        "one step at a time, and reports exactly what happened. **A machine-learning component** "
        "then tries to *guess* whether a machine will halt within L steps without running it that "
        "long; every guess is checked against the simulator. Nothing here claims to solve the "
        "general halting problem, which is undecidable."
    )

    st.subheader("The simulator always gives one of three answers")
    o1, o2, o3 = st.columns(3)
    o1.success("**Halted**\n\nReached the halt state H (or had no rule to apply). Proven.")
    o2.info("**Provably never halts**\n\nA full configuration (state, head, tape) repeated, so it "
            "is stuck in a cycle forever. Proven.")
    o3.warning("**Not halted within L**\n\nThe step budget ran out. **Unknown**: it may halt later "
               "or never. This is *not* 'never halts'.")

    summ = load_csv("results_summary.csv")
    rule_acc = summary_value(summ, "random", RULE_MODEL, "-", "accuracy")
    rf_f1 = summary_value(summ, "hard", "Random Forest", PROBE_FS, "f1")
    rf_rec = summary_value(summ, "hard", "Random Forest", PROBE_FS, "recall")
    if None not in (rule_acc, rf_f1, rf_rec):
        st.subheader(f"Headline results (L = {REPORT_L}, mean over 5 test splits)")
        h1, h2, h3 = st.columns(3)
        h1.metric(f"Random machines: just simulate {PROBE_STEPS} steps", f"{rule_acc:.1%} accurate",
                  help="On ordinary random machines a short simulation already answers almost "
                       "everything, so ML adds little.")
        h2.metric("Hard machines: F1, baseline → ML",
                  f"0.00 → {rf_f1:.2f}",
                  help=f"On machines still running after {PROBE_STEPS} steps the short simulation "
                       "finds none of the halters; the Random Forest finds most of them.")
        h3.metric("Slow halters found by the Random Forest", f"{rf_rec:.0%}",
                  help="Recall: the share of machines that really halt within L that the model flags.")

    st.subheader("Where each requirement is shown")
    st.markdown(
        "| Requirement (from the brief) | Where | What you see |\n"
        "|---|---|---|\n"
        "| Configurable simulator: states, tape symbols, rules, input | **Sidebar** | Pick a machine or "
        "edit its rules, start state, blank symbol and input. |\n"
        "| Tape operations, state transitions, halting states, step counting | **▶ Simulator** | Tape "
        "view, the rule applied at each step, the transition table δ, the state diagram, the full "
        "execution trace and the step counter. |\n"
        "| Dataset of machine descriptions and bounded execution traces | **📊 Experiments → Dataset** | "
        f"18,000 generated (machine, input) pairs with their rules, a {PROBE_STEPS}-step trace and the "
        "true outcome up to 1,000 steps. |\n"
        "| Features from transition tables and execution behaviour | **🔮 Halting prediction → "
        "features** | 19 rule-table features and 8 trial-run features, each explained, with how much "
        "the model relies on it. |\n"
        "| Classifiers: Decision Tree, Random Forest, neural network | **🔮 Halting prediction** | "
        "All three models give their guess for the current machine. |\n"
        "| Compare predictions with simulation at multiple step limits | **🔮 Halting prediction** | "
        "Per-machine table for L = 10 … 1000, and a batch test on fresh machines. |\n"
        "| How prediction quality changes with step limit and machine complexity | **📊 Experiments** | "
        "'Step limit' and 'Machine size' sections. |\n"
        "| Halting problem, decidability, computability | **📚 Theory notes** | Why 'halts?' is "
        "undecidable but 'halts within L?' is decidable. |\n"
        "| Never confuse 'halts within L' with 'has not halted within L' | **Everywhere** | Label 0 "
        "always means 'not halted within L', never 'never halts'. |\n"
    )

    st.subheader("Suggested 5-minute demo")
    st.markdown(
        "1. Sidebar → **4-state busy beaver** → ▶ Simulator: press *Next ▶* a few times, then *End ⏭* "
        "(halts at step 107).\n"
        "2. **Ping-pong** → *Provably never halts*: a configuration repeated.\n"
        "3. **Run-away right** → *Not halted within L*: it never halts, but no configuration repeats, "
        "so the honest answer is 'unknown'.\n"
        "4. Back to the 4-state busy beaver → 🔮 Halting prediction: simulation says 'not within L' "
        "up to L = 100 and 'halts' from L = 250; compare the three models.\n"
        "5. 🔮 Batch test (hard, Random Forest, L = 250): the model finds far more halters than the "
        f"{PROBE_STEPS}-step rule.\n"
        "6. 📊 Experiments: how quality changes with the step limit and the machine size."
    )

# --------------------------------------------------------------- simulator
with tab_sim:
    res = run(tm, input_str, max_steps, record=True, exact_loop_check=True)
    kind, msg = describe(res, max_steps)
    getattr(st, kind)(msg)

    trace = res.trace
    last = len(trace) - 1
    sim_id = hashlib.md5(
        f"{rules_text}|{input_str}|{max_steps}|{tm.start}|{tm.blank}".encode()
    ).hexdigest()[:12]
    step_key = f"step::{sim_id}"
    st.session_state[step_key] = min(st.session_state.get(step_key, 0), last)

    def _go(v: int) -> None:
        st.session_state[step_key] = max(0, min(last, v))

    def _shift(d: int) -> None:
        _go(st.session_state[step_key] + d)

    nav = st.columns([1, 1, 1, 1, 4])
    nav[0].button("⏮ Start", on_click=_go, args=(0,), width="stretch")
    nav[1].button("◀ Prev", on_click=_shift, args=(-1,), width="stretch")
    nav[2].button("Next ▶", on_click=_shift, args=(1,), width="stretch")
    nav[3].button("End ⏭", on_click=_go, args=(last,), width="stretch")
    if last > 0:
        step = st.slider("Step through the run", 0, last, key=step_key)
    else:
        step = 0
    s_no, s_state, s_head, s_tape = trace[step]

    left_col, right_col = st.columns([1.3, 1.0])
    with left_col:
        st.markdown(f"**Step {s_no} of {res.steps}**", help="Tape snapshot after this many steps")
        st.markdown(tape_html(s_tape, s_head, s_state, blank=tm.blank), unsafe_allow_html=True)
        e_kind, e_msg = explain_step(tm, res, trace, step)
        if e_kind == "plain":
            with st.container(border=True):
                st.markdown("**What happens next:** " + e_msg)
        else:
            getattr(st, e_kind)(e_msg)
        if len(trace) >= 3000:
            st.caption("Trace recording is capped at 3000 snapshots; the verdict above covers the full run.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Steps executed", res.steps)
        c2.metric("Non-blank cells at end", len(res.tape))
        c3.metric("Final state", res.state)
        if res.status == "halted" and res.tape:
            st.code("Final tape: " + res.tape_string(tm.blank), language=None)
    with right_col:
        st.markdown("**State transition diagram** (current state highlighted)")
        if len(tm.transitions) <= 45:
            st.graphviz_chart(to_dot(tm, active_state=s_state), width="stretch")
        else:
            st.info(f"Diagram omitted for large transition tables ({len(tm.transitions)} rules).")

    st.markdown("**Transition table δ** — each cell is *write, move, next state*; the rule used "
                "at this step is highlighted.")
    active = (s_state, s_tape.get(s_head, tm.blank)) if s_state != HALT else None
    st.dataframe(delta_table(tm, active), width="stretch")

    with st.expander(f"Full execution trace ({len(trace) - 1} steps recorded)"):
        tdf = trace_table(tm, trace)
        st.caption("One row per step: the rule applied and the tape afterwards. This is the "
                   "bounded execution trace the dataset is built from.")
        st.dataframe(tdf, width="stretch", hide_index=True, height=300)
        st.download_button("Download trace as CSV", tdf.to_csv(index=False), file_name="trace.csv",
                           mime="text/csv")

# -------------------------------------------------------------- prediction
with tab_pred:
    bundle = load_models()
    if bundle is None:
        st.warning("No trained models found. Run `python experiment.py` first.")
    else:
        st.markdown(
            f"The models look only at the rule table and a **{PROBE_STEPS}-step trial run**, then "
            "*guess* whether the machine halts within **L** steps. The simulator then runs it for "
            "real and checks the guess."
        )
        ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.0, 1.0])
        L = ctrl1.select_slider("Step limit L", options=LIMITS, value=100)
        pop_choice = ctrl2.radio(
            "Models trained on",
            ["hard", "random"],
            format_func=POP_LABEL.get,
            help=f"'hard' = trained only on machines still unresolved after {PROBE_STEPS} steps "
                 "(where prediction is useful); 'random' = trained on general 2-6 state random machines.",
        )
        primary_model = ctrl3.selectbox("Primary model (for batch test)", MODEL_NAMES, index=1)

        feats = all_features(tm, input_str)
        X = np.array([[feats[c] for c in bundle["cols"]]])
        truth = run(tm, input_str, L, exact_loop_check=True)
        actual = truth.status == "halted"

        st.subheader(f"Guesses vs. simulation at L = {L}")
        cols_ui = st.columns(4)
        for idx, m_name in enumerate(MODEL_NAMES):
            p_m, guess_m = model_verdict(bundle["models"][pop_choice][m_name][L], X)
            with cols_ui[idx], st.container(border=True):
                st.markdown(f"**{m_name}**")
                st.markdown(f"Guess: **{'halts within L' if guess_m else 'not within L'}**")
                st.caption(f"P(halts within {L}) = {p_m:.0%}")
                st.markdown(":green[✓ matches simulation]" if guess_m == actual
                            else ":red[✗ disagrees with simulation]")
        with cols_ui[3]:
            st.markdown("**Ground truth (simulation)**")
            k, m = describe(truth, L)
            getattr(st, k)(m)

        if pop_choice == "hard" and (feats["probe_halted"] or feats["probe_looping"]):
            st.info(f"This machine is already settled by the {PROBE_STEPS}-step trial run. The 'hard' "
                    f"models were trained only on machines still running after {PROBE_STEPS} steps, so "
                    "they are not meant for it: switch to the 'random' models, or simply trust the "
                    "simulation.")
        elif pop_choice == "hard" and L == 10:
            st.info(f"By construction no 'hard' training machine halts within {PROBE_STEPS} steps, "
                    "so the hard-trained models always answer 'not within 10' here.")

        # Same machine, every step limit: where do the models agree with simulation?
        st.markdown("**This machine at every step limit**")
        full = run(tm, input_str, max(LIMITS), exact_loop_check=True)
        table, agree = [], {m: 0 for m in MODEL_NAMES}
        for lim in LIMITS:
            sim = run(tm, input_str, lim, exact_loop_check=True)
            row = {"L": lim, "simulation": {"halted": "halts within L", "looping": "provably never halts",
                                            "limit": "not halted within L"}[sim.status]}
            for m_name in MODEL_NAMES:
                p_m, g_m = model_verdict(bundle["models"][pop_choice][m_name][lim], X)
                ok = g_m == (sim.status == "halted")
                agree[m_name] += ok
                row[m_name] = f"{'halts' if g_m else 'not within L'} ({p_m:.0%}) {'✓' if ok else '✗'}"
            table.append(row)
        st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
        st.caption(
            "Agreement with simulation: " + ", ".join(f"{m} {agree[m]}/{len(LIMITS)}" for m in MODEL_NAMES)
            + (f". Simulation: this machine halts at step {full.steps}." if full.status == "halted" else ".")
        )

        with st.expander("Features fed to the models (what the models actually see)"):
            fi = bundle.get("feature_importance")
            imp = (fi[fi["population"] == pop_choice].set_index("feature")["importance"]
                   if fi is not None else pd.Series(dtype=float))
            for title, cols in [("From the rule table (no execution)", STATIC_COLS),
                                (f"From a {PROBE_STEPS}-step trial run", PROBE_COLS)]:
                st.markdown(f"**{title}**")
                st.dataframe(pd.DataFrame({
                    "feature": cols,
                    "value": [feats[c] for c in cols],
                    "meaning": [FEATURE_HELP.get(c, "") for c in cols],
                    "importance": [imp.get(c, np.nan) for c in cols],
                }), width="stretch", hide_index=True, column_config={
                    "value": st.column_config.NumberColumn(format="%.3g"),
                    "importance": st.column_config.ProgressColumn(
                        "importance (Random Forest)", min_value=0.0,
                        max_value=float(max(imp.max(), 0.01)) if len(imp) else 1.0, format="%.3f"),
                })

        st.divider()
        st.subheader("Batch test on fresh unseen machines")
        st.markdown(
            "Generates new random machines the models have never seen, asks the primary model to "
            f"guess, simulates each one to L = {L}, and scores the guesses. The "
            f"*{PROBE_STEPS}-step rule* (\"halts if it already halted in the trial run\") is the baseline."
        )
        b1, b2 = st.columns(2)
        n_batch = b1.slider("Number of machines", 50, 1000, 300, step=50)
        batch_pop = b2.radio(
            "Test generator",
            ["hard", "random"],
            index=0 if pop_choice == "hard" else 1,
            format_func=POP_LABEL.get,
            horizontal=True,
            key=f"batch_pop::{pop_choice}",
            help="Defaults to the same distribution the selected models were trained on.",
        )
        if batch_pop != pop_choice:
            st.warning(
                f"Distribution shift: models trained on **{pop_choice}** machines are being "
                f"tested on **{batch_pop}** machines. Expect degraded accuracy "
                "(see the out-of-distribution discussion in the report)."
            )
        if st.button("Run batch test"):
            rng = random.Random(987654)  # differs from the training seeds -> unseen machines
            clf = bundle["models"][pop_choice][primary_model][L]
            rows = []
            while len(rows) < n_batch:
                m = sample_machine(rng, batch_pop)
                i = random_input(rng)
                if batch_pop == "hard" and run(m, i, PROBE_STEPS).status != "limit":
                    continue  # hard population = unresolved after the probe
                f = all_features(m, i)
                xp = np.array([[f[c] for c in bundle["cols"]]])
                truth_b = int(run(m, i, L).status == "halted")
                rows.append((truth_b, int(clf.predict(xp)[0]), int(f["probe_halted"])))
            d = pd.DataFrame(rows, columns=["actual", "ml", "rule"])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"{primary_model} accuracy", f"{accuracy_score(d.actual, d.ml):.1%}")
            m2.metric(f"{primary_model} F1", f"{f1_score(d.actual, d.ml, zero_division=0):.3f}")
            m3.metric(f"Rule ({PROBE_STEPS} steps) accuracy", f"{accuracy_score(d.actual, d.rule):.1%}")
            m4.metric(f"Rule ({PROBE_STEPS} steps) F1", f"{f1_score(d.actual, d.rule, zero_division=0):.3f}")
            st.caption(
                f"{primary_model}: precision {precision_score(d.actual, d.ml, zero_division=0):.3f}, "
                f"recall {recall_score(d.actual, d.ml, zero_division=0):.3f}; "
                f"{int(d.actual.sum())} of {len(d)} machines actually halt within {L} steps."
            )

            st.write(f"Confusion matrix ({primary_model}):")
            st.dataframe(
                pd.crosstab(
                    d.actual.map({0: "actual: not within L", 1: "actual: halts"}),
                    d.ml.map({0: "predicted: not within L", 1: "predicted: halts"}),
                ),
                width="stretch",
            )

# ------------------------------------------------------------- experiments
SECTIONS = ["📦 Dataset", "📈 Step limit", "🧩 Machine size", "🔍 Features & probe",
            "🛡️ Robustness", "⏱️ Cost", "📋 All numbers"]
FIGURES = {
    "📈 Step limit": [
        ("accuracy_vs_limit.png", "Prediction quality vs step limit L (mean ± std over 5 seeds)",
         "Top: on random machines everything is flat near 98% because almost every halter stops within "
         f"{PROBE_STEPS} steps, so the {PROBE_STEPS}-step rule (grey dashed) is already as good as the "
         "models. Bottom: on hard machines both baselines score F1 = 0; the models find real signal. "
         "A larger L can only turn 'not within L' labels into 'halts', so once no new machine halts "
         "the curves stop changing."),
        ("hard_cases.png", "Hard machines in detail: F1, PR-AUC and how many machines halt by each L",
         "Right panel: the number of slow halters stops growing after L ≈ 100-250, which is why "
         "results for L = 250, 500 and 1000 are identical."),
    ],
    "🧩 Machine size": [
        ("accuracy_vs_complexity.png", "Trained on small machines, tested on bigger unseen ones",
         "Without running the machine (grey dashed), accuracy drops as machines get bigger. With the "
         "trial run, the tree models keep working on sizes they never saw; the neural network's "
         "guesses collapse on 9-10-state machines because it extrapolates linearly."),
    ],
    "🔍 Features & probe": [
        ("feature_importance.png", "Which features the Random Forest relies on",
         "Random machines: 'did it already halt in the trial run?' dominates. Hard machines: whether "
         "H can be reached in the state diagram, plus whether the head drifts away (a run-away)."),
        ("probe_sweep.png", "Effect of a longer trial run (k steps)",
         "Going from 10 to 25 trial steps helps more than any model: the trial run itself starts to "
         "see the halts. More simulation beats a smarter guess."),
    ],
    "🛡️ Robustness": [
        ("mlp_ablation.png", "Fixing the neural network on imbalanced data",
         "Only ~8% of hard machines halt. Unweighted, the network misses most of them; tuning its "
         "decision threshold on a validation split gives the best recall and F1."),
        ("ood_heatmap.png", "Hand-written machines the models never saw (busy beavers etc.)",
         "All models correctly reject the three non-halters. Random-trained models miss slow "
         "halters; hard-trained trees catch the 4-state busy beaver and the countdown."),
    ],
}

with tab_exp:
    res_df = load_csv("results_summary.csv")
    if res_df is None:
        st.warning("No results yet. Run `python experiment.py`.")
    else:
        st.markdown(
            "Results of `experiment.py`: 3 models × 2 feature sets × 7 step limits × 5 train/test "
            "splits on two machine populations. Pick a section:"
        )
        section = st.radio("Section", SECTIONS, horizontal=True, key="exp_section",
                           label_visibility="collapsed")

        if section == "📦 Dataset":
            st.subheader("The dataset: machine descriptions and bounded execution traces")
            st.markdown(
                "Every row is one *(machine, input)* pair: the machine's rules, the input, a "
                f"{PROBE_STEPS}-step trace, and the **true outcome from simulating up to 1,000 steps**. "
                "So every label is exact, never a guess. The **random** population is small machines "
                "(2-6 states); the **hard** population keeps only pairs still running after "
                f"{PROBE_STEPS} steps, where a cheap simulation has already failed."
            )
            ds_pop = st.radio("Population", ["random", "hard"], horizontal=True, key="ds_pop",
                              format_func=POP_LABEL.get)
            ds = load_csv(f"dataset_{ds_pop}.csv")
            if ds is None:
                st.warning(f"results/dataset_{ds_pop}.csv not found.")
            else:
                vc = ds["status"].value_counts()
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Pairs", f"{len(ds):,}")
                d2.metric("Halted", f"{vc.get('halted', 0):,}")
                d3.metric("Provably looping", f"{vc.get('looping', 0):,}")
                d4.metric("Unknown after 1,000 steps", f"{vc.get('limit', 0):,}")

                halters = ds.loc[ds["halt_step"] >= 0, "halt_step"]
                edges = [0] + LIMITS
                spans = list(zip(edges, edges[1:]))  # (0, 10], (10, 25], ... ; step 0 counts in the first
                brackets = pd.DataFrame({
                    "halts between steps": [f"{a + 1 if a else 0}-{b}" for a, b in spans],
                    "machines": [int(((halters > a) | (a == 0)).mul(halters <= b).sum()) for a, b in spans],
                })
                brackets["share of halters (%)"] = 100 * brackets["machines"] / max(len(halters), 1)
                st.markdown("**When do the halting machines halt?** (the step-limit brackets used in the study)")
                st.dataframe(brackets, width="stretch", hide_index=True, column_config={
                    "share of halters (%)": st.column_config.ProgressColumn(
                        min_value=0.0, max_value=100.0, format="%.1f%%"),
                })
                st.markdown("**Sample rows** (rules use the same text format as the sidebar, `;`-separated; "
                            "trace = `step:state@head`)")
                st.dataframe(
                    ds.sample(min(15, len(ds)), random_state=0)[
                        ["rules", "input_str", "probe_trace", "status", "halt_step"]],
                    width="stretch", hide_index=True, column_config={
                        "input_str": "input",
                        "probe_trace": f"first {PROBE_STEPS} steps",
                        "halt_step": st.column_config.NumberColumn("halt step (-1 = did not halt)"),
                    })
                st.caption(f"Full file: results/dataset_{ds_pop}.csv ({len(ds.columns)} columns, "
                           "including all 27 features).")

        elif section in FIGURES:
            for fname, title, takeaway in FIGURES[section]:
                if (RESULTS / fname).exists():
                    st.subheader(title)
                    st.image(str(RESULTS / fname))
                    st.info("**Takeaway:** " + takeaway)

        elif section == "⏱️ Cost":
            st.subheader("Cost per machine: simulating L steps vs ML triage")
            cost = load_csv("cost.csv")
            if cost is None:
                st.warning("results/cost.csv not found.")
            else:
                st.dataframe(cost[["method", "ms_per_pair"]], width="stretch", hide_index=True,
                             height=36 * (len(cost) + 1) + 3,
                             column_config={"ms_per_pair": st.column_config.NumberColumn(
                                 "milliseconds per pair", format="%.4f")})
                feat = cost.loc[cost["method"].str.startswith("features"), "ms_per_pair"]
                rf = cost.loc[cost["method"].str.contains("Random Forest"), "ms_per_pair"]
                sims = cost[cost["method"].str.startswith("simulate")]
                if len(feat) and len(rf):
                    triage = float(feat.iloc[0] + rf.iloc[0])
                    cheaper = sims[sims["ms_per_pair"] > triage]
                    st.info(
                        f"**Takeaway:** features + a Random Forest guess cost about {triage:.3f} ms per "
                        "machine, whatever L is, while simulation cost grows with L. "
                        + (f"Here the guess is cheaper than simulating once L ≥ {int(cheaper['limit'].min())}. "
                           if len(cheaper) else "")
                        + "Every simulation in this project is under a millisecond, so ML only pays off "
                          "for huge budgets or very many machines; it is a triage tool, not a replacement."
                    )

        else:  # All numbers
            st.subheader("Results table (mean ± std over 5 test splits)")
            e1, e2 = st.columns(2)
            pop = e1.radio("Population", ["random", "hard"], horizontal=True, key="exp_pop",
                           format_func=POP_LABEL.get)
            lim = e2.select_slider("Step limit", options=LIMITS, value=REPORT_L, key="exp_L")
            show = res_df[(res_df["population"] == pop) & (res_df["limit"] == lim)]
            metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
            table = show[["features", "model"]].copy()
            for mcol in metric_cols:
                table[mcol] = [f"{a:.3f} ± {0 if pd.isna(b) else b:.3f}" if not pd.isna(a) else "n/a"
                               for a, b in zip(show[f"{mcol}_mean"], show[f"{mcol}_std"])]
            table["test positives"] = (show["n_pos_test"].round(0).astype(int).astype(str) + " / "
                                       + show["n_test"].round(0).astype(int).astype(str))
            st.dataframe(table, width="stretch", hide_index=True)
            st.caption("F1 and PR-AUC matter most on the hard population: only ~8% of it halts, so "
                       "answering 'no' every time is already ~92% accurate but useless.")

            ood_df = load_csv("ood_benchmarks.csv")
            if ood_df is not None:
                with st.expander("Out-of-distribution table (all step limits)"):
                    st.dataframe(ood_df.round(3), width="stretch", hide_index=True)
            cx_df = load_csv("complexity.csv")
            if cx_df is not None:
                with st.expander("Complexity generalisation table (per seed)"):
                    st.dataframe(cx_df.round(3), width="stretch", hide_index=True)
            if (RESULTS / "summary.md").exists():
                with st.expander("All key numbers (results/summary.md)"):
                    st.markdown((RESULTS / "summary.md").read_text(encoding="utf-8"))

# ------------------------------------------------------------------ theory
with tab_theory:
    st.markdown(
        """
### Two questions that look alike

| | **HALT**: does M halt on w? | **HALT≤L**: does M halt on w within L steps? |
|---|---|---|
| Decidable? | **No** (Turing, 1936) | **Yes** |
| How to answer | No algorithm works for every machine | Simulate at most L steps and look |
| In this project | Never claimed | What the simulator answers exactly and the models guess |

### Why "does it halt?" is undecidable (sketch)
Suppose a program **D** could always answer "does M halt on w?". Build a machine **X** that, given a
machine M, asks D "does M halt on its own description?" and then does the **opposite**: it loops
forever if D says "halts" and halts if D says "loops". Now run X on itself. If X halts, D said it
loops; if X loops, D said it halts. Either way D is wrong, so D cannot exist.

### The three outcomes the simulator reports
- **Halted**: reached the halt state `H` (or met a missing rule) within the step budget.
- **Provably never halts**: a full configuration `(state, head position, tape)` repeated. The machine
  is deterministic, so it will repeat the same cycle forever.
- **Not halted within L**: unknown. It may halt after step L, or run forever without ever repeating a
  configuration (e.g. writing `1`s endlessly to the right). Detecting *every* non-halter would solve
  the halting problem, so this "unknown" answer is unavoidable.

### Where machine learning fits
Rule-table features (how reachable `H` is, run-away rules, …) plus a short trial run let Decision
Trees, Random Forests and neural networks **guess** which unresolved machines will halt soon. That is
useful for deciding which machines deserve a longer simulation, but a guess is never a proof.
        """
    )
    st.warning(
        "**What this app never claims:** that a machine *never* halts unless a configuration repeated "
        "(a proof), or that a model's guess settles anything. The ML label 0 means "
        "'has not halted within L steps', never 'never halts'."
    )
