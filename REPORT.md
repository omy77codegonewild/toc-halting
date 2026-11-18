# Turing Machine Simulation and Bounded Halting Prediction

**Theory of Computation — Problem 6 · Experimental Report**

**Live interactive app:** [https://toc-halting.streamlit.app](https://toc-halting.streamlit.app)

---

## Abstract

We build a configurable single-tape Turing machine (TM) simulator and a machine-learning (ML)
component that predicts whether a `(machine, input)` pair **halts within a step limit L**.
The simulator reports one of three honest outcomes: *halted*, *provably looping* (a full
configuration repeated) or *limit reached* (unknown). It never claims that a machine "does not
halt" unless that has been proved. We generate 6,000 random pairs and 12,000 "hard" pairs that are still
unresolved after a 10-step probe run, extract 19 static transition-table features and 8 probe
(execution-behaviour) features, and train a Decision Tree, a Random Forest and a neural network
(MLP). Every result is the mean ± standard deviation over 5 stratified train/test splits.

On unfiltered random machines, simply simulating 10 steps is already 97.8% accurate at L = 250,
so ML adds nothing there; static features alone reach 81.1% (Random Forest) without running the
machine at all. On the hard machines, both baselines score F1 = 0, while a Random Forest with
static + probe features reaches **F1 0.527 ± 0.008, recall 0.843, ROC-AUC 0.940**. That makes
ML a useful *triage filter* for deciding which unresolved machines deserve a longer simulation.
Raising the probe budget from 10 to 25 steps lifts F1 to about 0.91. Every prediction is
compared with ground truth obtained by simulation, which remains the only source of truth.

---

## 1. Background

### 1.1 Turing machines

A deterministic single-tape Turing machine is a tuple **M = (Q, Σ, Γ, δ, q₀, ⊔, H)**:

| Symbol | Meaning | In this project |
|---|---|---|
| Q | finite set of states | any names, e.g. `q0, q1, …` |
| Γ | tape alphabet, contains the blank ⊔ | any single characters; blank configurable (default `_`) |
| Σ ⊆ Γ \ {⊔} | input alphabet | `0`, `1` for generated machines; anything for hand-written ones |
| δ : Q × Γ ⇀ Γ × {L, R, S} × (Q ∪ {H}) | partial transition function | one rule per line: `state read write move next` |
| q₀ | start state | `q0` by default, configurable |
| H | halting state | `H` |

A **configuration** is the triple *(current state, head position, tape contents)*. One step
reads the symbol under the head, writes a symbol, moves the head left, right, or stays, and changes state.
If δ has no rule for the current *(state, symbol)*, the machine halts. This is the standard
convention for a partial δ.

### 1.2 The halting problem is undecidable

HALT = { ⟨M, w⟩ : M halts on input w } is **undecidable** (Turing, 1936). Sketch: suppose a
decider D(⟨M, w⟩) existed. Build the machine X that, on input ⟨M⟩, runs D(⟨M, ⟨M⟩⟩) and does the
opposite: it loops forever if D says "halts", and halts if D says "loops". Running X on ⟨X⟩ gives a
contradiction either way, so D cannot exist. Consequently **no simulator, feature set or neural
network can decide halting for all machines**. Anything that claims to is wrong on some input.

### 1.3 Bounded halting is decidable

HALT≤L = { ⟨M, w, L⟩ : M halts on w within L steps } **is decidable**: simulate at most L steps
and answer. This costs O(L) steps. The two questions differ in exactly one way:

> "has **not** halted within L steps" ≠ "**never** halts".

A machine may halt at step L + 1. This project's ML label is therefore always
**1 = halts within L** versus **0 = has not halted within L**, and the report never interprets 0
as "never halts".

### 1.4 When can non-halting be proved?

A deterministic machine whose configuration *repeats* is trapped in a cycle forever: from the
same configuration it must make exactly the same moves again. Detecting a repeated configuration is
therefore a **sound proof of non-halting**. It is not complete. A machine that keeps writing
new cells (e.g. `q0 _ 1 R q0`, or a binary counter) never repeats a configuration, never halts,
and can only be reported as *limit reached*.

---

## 2. Method

### 2.1 Simulator design ([tm.py](tm.py))

* **Sparse, two-way infinite tape:** a Python `dict[int, str]` stores only non-blank cells, so
  negative positions and unbounded growth need no special handling.
* **Configurable:** states, tape symbols (any single characters), blank symbol, start state,
  transition rules (`L`/`R`/`S` moves) and the input string. The parser reports line-numbered errors
  (wrong field count, bad move, multi-character symbol, duplicate rule).
* **Three outcomes**, exactly one per run:

| Status | Meaning | Guarantee |
|---|---|---|
| `halted` | reached `H` or had no applicable rule within the budget | proven to halt at step t ≤ L |
| `looping` | a full configuration (state, head, tape) repeated | **proven never to halt** |
| `limit` | budget exhausted | **unknown**: may halt later or run forever |

* **Loop detection:** two interchangeable detectors.
  * *Exact:* store every configuration as `(state, head, frozenset(tape))`. This is a definite
    proof, but costs O(tape size) per step. The app uses it.
  * *Zobrist hashing* (fast, used for dataset generation): keep a 64-bit hash of the tape that is
    updated in O(1) per step by XOR-ing out the old cell's hash and XOR-ing in the new one. A
    repeated `(state, head, hash)` signals a loop. A false alarm requires a 64-bit hash collision
    (birthday bound ≈ n²/2⁶⁵ ≈ 3·10⁻¹⁴ for a 1,000-step run). The exact detector is available
    whenever a definite proof is needed. Tests check that both detectors agree on the presets.
* **Trace recording** for the step-through UI (capped at 3,000 snapshots; the verdict always
  covers the full run) and a Graphviz DOT renderer for the state diagram.

### 2.2 Datasets ([experiment.py](experiment.py))

Machines use the alphabet {`_`, `0`, `1`} with a random rule for every *(state, symbol)*.
Each rule goes to `H` with probability `p_halt`, drawn per machine, so that labels stay mixed. Inputs
are random binary strings of length 0–8. Pairs are de-duplicated. The ground truth for each pair is
obtained by simulating up to **1,000 steps** and storing `status` and `halt_step`. Because of
this, the labels are **exact** for every L ≤ 1,000. No label in the dataset is a guess.

| Population | Generator | Pairs | Halted | Looping | Limit (1000) | Halts within 250 |
|---|---|---|---|---|---|---|
| `random` | 2–6 states, p_halt ∈ [0.03, 0.30] | 6,000 | 3,315 | 326 | 2,359 | 3,315 (55.2%) |
| `hard` | 4–10 states, p_halt ∈ [0.01, 0.12], 25% allow `S` moves, **kept only if still unresolved after the 10-step probe** | 12,000 | 1,004 | 625 | 10,371 | 1,004 (8.4%) |

The hard population is the interesting one: a cheap simulation has already failed on these
pairs, which is exactly where a predictor could save work. The datasets store the rules, the input
and a 10-step trace preview for every pair (`results/dataset_*.csv`).

### 2.3 Features ([features.py](features.py))

* **19 static features** (no execution): number of states/rules, alphabet size; fractions of
  rules that halt, move right, stay, write blank, or are self-loops; structural motifs (right/left
  "sweepers", "run-away writers"); a breadth-first search over the state graph giving the number of
  reachable states, the shortest distance from the start state to `H`, and the number of reachable halting rules;
  plus input statistics (length, non-blank cells, distinct symbols, number of 1s).
* **8 probe features** from a 10-step trial run: halted / looping flags, steps executed,
  distinct states visited, head span, net head movement, non-blank cells and tape growth.

### 2.4 Models and class imbalance ([ml_models.py](ml_models.py))

| Model | Settings | Imbalance handling |
|---|---|---|
| Decision Tree | max depth 8 | `class_weight="balanced"` |
| Random Forest | 120 trees, max depth 14 | `class_weight="balanced"` |
| Neural Net (MLP) | standardised inputs, hidden layers 64-32, 60 epochs (Adam) | **decision threshold tuned on a validation split** |

Scikit-learn's `MLPClassifier` has no class weights, and only 8.4% of hard pairs are positive.
Inside `fit`, the training fold is split 80/20 (stratified). The network trains on the 80% part,
and the probability threshold that maximises F1 on the 20% part replaces the default 0.5. The test
fold is never touched. While developing this, we found that scikit-learn's built-in *early
stopping* made the MLP unstable on imbalanced data. It monitors validation *accuracy*, which
plateaus at "predict everything 0" and stops training after a few epochs. In a 3,000-pair pilot this gave
F1 0.34 ± 0.15 across seeds, against 0.48 ± 0.03 with a fixed 60 epochs. We therefore train for a fixed 60 epochs. The
alternatives are compared in §3.3.

### 2.5 Evaluation protocol

* Step limits **L ∈ {10, 25, 50, 100, 250, 500, 1000}**, two feature sets (*static only* and
  *static + 10-step probe*), three models.
* **5 stratified 70/30 train/test splits** (seeds 42–46). Every number below is mean ± std over
  those 5 splits. Hard test sets contain 3,600 pairs with about 300 positives (§3.1).
* Metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC (average precision). PR-AUC and F1
  matter most for the imbalanced hard population, where predicting all-zeros gives 91.6% accuracy.
* Baselines: **majority class** and **"simulate 10 steps"** (predict *halts* iff the probe halted).
* Implementation note: when two step limits produce identical labels (e.g. nothing new halts
  between L = 250 and L = 1000), training would be identical, so the results are reused. The 5
  seeds are independent, so they run in parallel processes; this changes only the wall-clock time.

---

## 3. Experiments and results

All numbers come from `python experiment.py` (full run ≈ 3 minutes on a 12-core CPU). The
auto-generated `results/summary.md` contains every table below.

### 3.1 Prediction quality vs the step limit

![Accuracy / F1 vs step limit](results/accuracy_vs_limit.png)

Test-set positives (mean over seeds):

| Population | L=10 | L=25 | L=50 | L=100 | L=250 | L=500 | L=1000 |
|---|---|---|---|---|---|---|---|
| random (of 1,800) | 954 | 990 | 993 | 994 | 994 | 994 | 994 |
| hard (of 3,600) | 0 | 259 | 294 | 300 | 301 | 301 | 301 |

Static + 10-step probe features:

| Model | random accuracy, L=25 | random accuracy, L=250 | hard F1, L=25 | hard F1, L=50 | hard F1, L=250 |
|---|---|---|---|---|---|
| Rule: simulate 10 steps | 0.980 ± 0.004 | 0.978 ± 0.004 | 0.000 | 0.000 | 0.000 |
| Decision Tree | 0.978 ± 0.004 | 0.975 ± 0.003 | 0.440 ± 0.027 | 0.486 ± 0.014 | 0.491 ± 0.019 |
| Random Forest | 0.981 ± 0.005 | 0.978 ± 0.004 | 0.484 ± 0.014 | 0.528 ± 0.010 | 0.527 ± 0.008 |
| Neural Net (MLP) | 0.982 ± 0.001 | 0.978 ± 0.004 | 0.462 ± 0.025 | 0.509 ± 0.015 | 0.513 ± 0.019 |

**Observations.**
* **Random machines halt fast or not at all.** In the full dataset 3,186 of the 3,315 halters
  (96.1%) halt within 10 steps, and the slowest halts at step 87. That is why the 10-step rule is so strong, and
  why every curve is flat for L ≥ 100.
* **Hard machines:** at L = 10 there are no positives by construction (the population was
  filtered to be unresolved at step 10), so no model is trained there. From L = 25 the models
  find signal, and F1 rises slightly up to L = 50 as more slow halters become positive. The
  slowest hard halter halts at step 157 (87% halt by step 25), so L = 250, 500 and 1000 have identical
  labels and identical results.
* **The step limit therefore matters mainly through the labels:** a larger L can only turn
  zeros into ones. Once no new machine halts, prediction quality stops changing.

### 3.2 The hard cases in detail (L = 250)

![Hard cases](results/hard_cases.png)

| Features | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| — | Majority / 10-step rule | 0.916 | 0.000 | 0.000 | 0.000 | 0.500 | 0.084 |
| static only | Decision Tree | 0.735 ± 0.018 | 0.215 ± 0.009 | 0.817 ± 0.042 | 0.341 ± 0.011 | 0.820 ± 0.009 | 0.268 ± 0.008 |
| static only | Random Forest | 0.837 ± 0.013 | 0.289 ± 0.017 | 0.646 ± 0.024 | 0.399 ± 0.015 | 0.874 ± 0.003 | 0.340 ± 0.017 |
| static only | Neural Net (MLP) | 0.814 ± 0.046 | 0.259 ± 0.027 | 0.626 ± 0.158 | 0.359 ± 0.022 | 0.863 ± 0.005 | 0.306 ± 0.013 |
| static + probe | Decision Tree | 0.839 ± 0.012 | 0.335 ± 0.017 | **0.924 ± 0.021** | 0.491 ± 0.019 | 0.906 ± 0.007 | 0.408 ± 0.020 |
| static + probe | Random Forest | **0.874 ± 0.007** | 0.384 ± 0.012 | 0.843 ± 0.018 | **0.527 ± 0.008** | **0.940 ± 0.003** | **0.520 ± 0.015** |
| static + probe | Neural Net (MLP) | 0.873 ± 0.020 | 0.383 ± 0.037 | 0.794 ± 0.082 | 0.513 ± 0.019 | 0.932 ± 0.005 | 0.467 ± 0.027 |

Both baselines find **none** of the 301 slow halters. With static + probe features, the Random
Forest finds 84% of them and the Decision Tree 92%, while wrongly flagging only 12.4% (RF) and
16.8% (DT) of the non-halters. The positive rate is 8.4%, so PR-AUC 0.52 is about six times the chance level. Precision is
modest (≈ 0.38: roughly 6 of every 10 flagged machines do not halt by L). That is acceptable
for triage, where a flag only means "simulate this one longer", but it is far from a proof.
Probe features add 0.13–0.15 F1 for every model over static features alone.

### 3.3 Fixing the neural network on imbalanced data

![MLP ablation](results/mlp_ablation.png)

| Population | MLP variant | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| hard | unweighted, threshold 0.5 | 0.500 ± 0.041 | 0.397 ± 0.045 | 0.440 ± 0.031 | 0.481 ± 0.032 |
| hard | oversampled minority, threshold 0.5 | 0.425 ± 0.009 | 0.607 ± 0.017 | 0.500 ± 0.010 | 0.436 ± 0.017 |
| hard | **threshold tuned on validation** | 0.383 ± 0.037 | **0.794 ± 0.082** | **0.513 ± 0.019** | 0.467 ± 0.027 |
| random | unweighted, threshold 0.5 | 0.983 ± 0.007 | 0.977 ± 0.006 | 0.980 ± 0.002 | 0.999 |
| random | threshold tuned on validation | 0.989 ± 0.012 | 0.970 ± 0.013 | 0.979 ± 0.003 | 0.999 |

On the imbalanced hard data, the unweighted MLP misses 60% of the halters. Both fixes trade
precision for recall, and threshold tuning gives the best F1 and recall at well under half the training cost of
oversampling (2.6 s vs 6.5 s per fit). On the balanced random population no fix is needed and all variants tie.
There the tuned threshold is also unstable (0.32–0.97 across seeds), because the F1 curve is flat.

### 3.4 Generalising to bigger, unseen machines

Models are trained only on small machines (random: ≤ 4 states; hard: ≤ 6 states) and tested
on every size, including sizes never seen in training.

![Complexity](results/accuracy_vs_complexity.png)

| Population (L = 250) | Model | seen sizes | unseen sizes |
|---|---|---|---|
| random, static only | Random Forest accuracy | 0.847 (2 st.) → 0.817 (4 st.) | 0.794 (5 st.) → 0.765 (6 st.) |
| random, static + probe | Random Forest accuracy | 0.988 (2) → 0.978 (4) | 0.976 (5) → 0.965 (6) |
| hard, static + probe | Random Forest F1 / ROC-AUC | 0.444–0.520 / 0.942–0.967 | 0.512–0.558 / 0.888–0.940 |
| hard, static + probe | Decision Tree F1 | 0.401–0.446 | 0.496–0.525 |
| hard, static + probe | MLP F1 (recall) | 0.347–0.466 (0.47–0.52) | **0.360 → 0.051** (0.32 → 0.03) |

* Without execution, accuracy falls steadily with machine size: bigger machines are harder to
  judge from their rule table alone. The 10-step probe almost removes this effect on random machines.
* On hard machines the **tree models transfer well**: their F1 on unseen 7–10-state machines is
  as good as on the training sizes, and the Random Forest's ROC-AUC drops only from 0.94–0.97 (seen sizes) to
  0.89–0.94 (unseen sizes).
* The **MLP fails to extrapolate**. Its ranking stays reasonable (ROC-AUC 0.87–0.92 on unseen
  sizes), but its probabilities drift, so the tuned threshold misses almost every halter on
  9–10-state machines. Features such as `n_states` and `n_rules` fall outside the training range.
  Trees simply treat them as "the largest value seen", while the network extrapolates them linearly.

### 3.5 Which features matter?

![Feature importance](results/feature_importance.png)

Random Forest Gini importance (L = 250, mean over 5 seeds):

* **random:** `probe_halted` 0.343, `probe_steps` 0.201, `probe_head_span` 0.115,
  `dist_to_halt` 0.074, `halting_rules_reachable` 0.056. *Did it already halt?* dominates.
* **hard** (no machine halted in the probe): `halting_rules_reachable` 0.172,
  `dist_to_halt` 0.171, `frac_to_halt` 0.129, `probe_net_move` 0.125, `probe_head_span` 0.079,
  `probe_distinct_states` 0.059. The model combines *can the halt state be reached in the state
  graph, and how easily?* with *is the head drifting away (a run-away) or staying local?*

### 3.6 How long should the probe run be?

![Probe sweep](results/probe_sweep.png)

| probe k | DT F1 | RF F1 | MLP F1 | RF PR-AUC |
|---|---|---|---|---|
| 0 (static only) | 0.341 ± 0.011 | 0.399 ± 0.015 | 0.359 ± 0.022 | 0.340 ± 0.017 |
| 5 | 0.406 ± 0.013 | 0.466 ± 0.015 | 0.438 ± 0.012 | 0.431 ± 0.011 |
| 10 | 0.491 ± 0.019 | 0.527 ± 0.008 | 0.513 ± 0.019 | 0.520 ± 0.015 |
| 25 | 0.815 ± 0.026 | 0.906 ± 0.011 | 0.918 ± 0.013 | 0.978 ± 0.002 |
| 50 | 0.959 ± 0.016 | 0.988 ± 0.003 | 0.987 ± 0.002 | 0.996 ± 0.002 |

The jump between k = 10 and k = 25 is not the models becoming clever. The probe itself starts to
*see* the halts: 259 of the 301 test halters halt by step 25 (§3.1), so `probe_halted` almost
gives away the label. At k = 50 the "prediction" is essentially simulation. This is the central
trade-off: **every step of extra simulation buys more accuracy than any model can**, and ML is
valuable only for the part of the budget you cannot afford to simulate.

### 3.7 Out-of-distribution: busy beavers and hand-written machines

The final models (trained on all data) were applied to 11 hand-written machines that look
nothing like random machines. Cells show the predicted P(halts within 250); ✗ marks a wrong
yes/no decision.

![OOD heatmap](results/ood_heatmap.png)

| Machine | Truth (steps) | rand DT | rand RF | rand MLP | hard DT | hard RF | hard MLP |
|---|---|---|---|---|---|---|---|
| Binary incrementer | halted (8) | 1.00 | 0.98 | 1.00 | 0.86 | 0.45 ✗ | 0.02 ✗ |
| 2-state busy beaver | halted (6) | 1.00 | 0.96 | 1.00 | 0.00 ✗ | 0.19 ✗ | 0.00 ✗ |
| 3-state busy beaver | halted (14) | 0.02 ✗ | 0.01 ✗ | 0.00 ✗ | 0.32 ✗ | 0.07 ✗ | 0.00 ✗ |
| 4-state busy beaver | halted (107) | 0.21 ✗ | 0.07 ✗ | 0.09 ✗ | 0.70 | 0.63 | 0.01 ✗ |
| Unary addition 111+11 | halted (8) | 1.00 | 0.97 | 1.00 | 0.86 | 0.54 | 0.18 |
| Binary palindrome checker | halted (36) | 0.73 | 0.46 ✗ | 0.99 | 0.95 | 0.47 ✗ | 0.94 |
| Binary countdown to zero | halted (125) | 0.08 ✗ | 0.29 ✗ | 0.97 ✗ | 0.86 | 0.51 | 0.00 ✗ |
| Custom symbols a/b flip | halted (6) | 1.00 | 1.00 | 1.00 | 0.86 | 0.43 ✗ | 0.00 ✗ |
| Ping-pong | looping (2) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Run-away right | limit | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Binary counter (forever) | limit | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

(The random-trained MLP gives the countdown 0.97 but still answers "no": its tuned threshold at
L = 250 is 0.972, see §3.3.) Accuracy over all 11 machines and all 7 step limits:
random-trained DT 83%, MLP 83%, RF 78%; hard-trained DT 71%, RF 58%, MLP 53%.

* **All six models correctly reject all three non-halters**, including the two that are not provable by
  repetition.
* **Random-trained models handle fast halters (≤ 10 steps)** but miss slow ones (BB(3), BB(4), the
  countdown). They learned that "not halted after 10 steps" means "won't halt soon".
* **Hard-trained trees catch slow halters** (BB(4) at step 107, the countdown at step 125), though none catches BB(3). The
  hard models were never shown a machine that halts within 10 steps, so they misjudge some fast halters.
* **Practical pipeline:** simulate 10 steps first. If the machine is still unresolved, ask the
  hard-trained tree models. This mirrors how the hard population was built.

### 3.8 Cost: simulation vs ML triage

Average wall-clock time per hard `(machine, input)` pair, measured on 1,000 hard pairs in the
same run (`results/cost.csv`; absolute times depend on the computer):

| Method | ms per pair |
|---|---|
| simulate L = 10 / 100 / 250 steps | 0.018 / 0.103 / 0.231 |
| simulate L = 500 / 1,000 steps | 0.472 / 0.924 |
| compute features (static + 10-step probe) | 0.067 |
| predict, batched: Decision Tree / Random Forest / MLP | 0.0005 / 0.036 / 0.002 |

Simulation cost grows linearly with L, because unresolved machines use the whole budget. ML triage
costs a fixed amount: features plus a Random Forest prediction take 0.103 ms, the same as simulating
100 steps. **Triage is only worth it when L is much larger than about 100 steps.** At L = 1,000 it is
about 9× cheaper than simulating, and the advantage grows with L. In this project every
simulation finishes in under a millisecond, so ML saves nothing in practice. The argument
applies to budgets of millions of steps or to very large batches of machines.

---

## 4. Discussion

**When plain simulation wins.** Whenever the budget can simply be simulated. Simulation is exact,
and on random machines 10 steps already resolves about 98% of the bounded question. Even on hard
machines, extending the probe from 10 to 25 steps (§3.6) beats any model. ML must never replace
simulation when an exact answer is affordable.

**When ML helps.** When L is far larger than what can be simulated for every candidate, and the
goal is to *rank* or *filter*: e.g. choosing which of thousands of unresolved machines deserve a
long run. Here a Random Forest flags 84% of the slow halters while discarding most non-halters,
and the trees transfer to bigger machines than they were trained on.

**Limitations.**
1. **Random machines are not real programs.** No generated machine halts after step 157, so
   step limits beyond that cannot be studied with this generator. Structured slow halters (busy
   beavers, counters) are rare in random populations, and the OOD results show the models are
   sensitive to such shifts.
2. **Low precision.** About 6 in 10 machines flagged on the hard population do not halt by L. This
   is acceptable for triage and unusable as evidence about any single machine.
3. **Distribution dependence.** Random-trained and hard-trained models disagree on fast vs slow
   halters (§3.7). The MLP does not extrapolate to bigger machines (§3.4).
4. **Loop detection is sound but incomplete.** Machines that diverge without repeating a
   configuration (counters, run-aways) are only ever reported as *limit reached*, by design.
5. **Undecidability.** No amount of data removes the theoretical barrier of §1.2. A machine that
   searches for a counterexample to an open conjecture halts exactly when the conjecture is false,
   and no feature of its rule table can know that.

---

## 5. Conclusion

We built a configurable Turing machine simulator that keeps three outcomes apart: *halted*,
*provably looping* and *limit reached*. We used it to study whether ML can predict
*halting within L steps*. Prediction is easy where it is least needed: random machines halt quickly, and a
10-step simulation is 97.8% accurate. On machines that a short simulation cannot resolve, simple
baselines fail completely (F1 = 0), while tree ensembles using transition-graph and probe-behaviour
features recover 84–92% of the slow halters (best F1 0.527, ROC-AUC 0.940) and generalise to larger
machines. The models are a heuristic triage tool. Every prediction in this project is checked
against simulation, which stays the ground truth, and nothing here claims to decide the general
halting problem.

---

## 6. Reproducibility

```bash
pip install -r requirements.txt
python -m pytest tests -v     # 37 tests: simulator, features, models, headless app
python experiment.py          # regenerates everything in results/ (≈ 3 min)
streamlit run app.py          # interactive simulator and predictor
```

The same app is deployed at [https://toc-halting.streamlit.app](https://toc-halting.streamlit.app)
(free hosting, so the first load after a quiet spell can take about 30 s while it wakes up).

All randomness is seeded (dataset seeds 42 and 1042, split seeds 42–46), so a re-run reproduces
every metric in this report. Only the timing columns and §3.8 vary between machines.
