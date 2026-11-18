# Viva notes: Turing Machine Simulation and Bounded Halting Prediction

Plain-language notes for the oral exam. The full details are in [REPORT.md](REPORT.md).

---

## 1. The project in one paragraph

We wrote a program that runs Turing machines step by step. For any machine and input, it says
one of three things: **it halted**, **it is provably stuck in a loop forever**, or **it ran out of
steps, so we don't know**. We then trained machine-learning models to *guess* whether a machine
will halt **within L steps**, without running it that long. The guesses are compared with the
real answer from the simulator. We never claim to solve the halting problem, because it is
impossible.

---

## 2. What each file does

| File | In simple words |
|---|---|
| `tm.py` | **The simulator.** Stores a machine as a table `(state, symbol) → (write, move, next state)`. `run()` executes it for at most L steps and returns `halted` / `looping` / `limit`. Also: the text parser for rules, the 11 example machines (busy beavers, incrementer, palindrome checker…), the random machine generator, and the state-diagram drawing (Graphviz DOT). |
| `features.py` | **Turns a machine into numbers** for ML. 19 *static* features read from the rule table (e.g. "what fraction of rules go to H?", "how many steps from the start state to H in the state graph?") and 8 *probe* features from running just 10 steps (e.g. "did it halt?", "how far did the head move?"). |
| `ml_models.py` | **The three classifiers:** Decision Tree, Random Forest, and a neural network (`BalancedMLP`). The MLP picks its own decision threshold on a validation split, because only about 8% of the hard examples are positive. |
| `experiment.py` | **Runs every experiment.** It generates the datasets, trains and tests the models over 5 random splits at 7 step limits, runs the extra studies (machine size, feature importance, probe length, MLP fix, hand-written machines, cost), and writes everything to `results/` (CSVs, plots, `summary.md`). |
| `app.py` | **The interactive website** (Streamlit). Five tabs. **Overview**: which part of the app shows each requirement of the brief. **Simulator**: edit a machine and step through the run (Prev/Next buttons), watching the tape, the rule applied at each step, the transition table and the state diagram with the current state highlighted. **Halting prediction**: what the three models guess vs. what simulation says, at every L, the features explained, plus a batch test. **Experiments**: the dataset, and every plot with a one-line takeaway. **Theory notes**. |
| `tests/` | **37 automatic checks:** the incrementer gives the right answers, the busy beavers write the right number of 1s, ping-pong is detected as looping, the run-away gives "limit", bad rules are rejected, a missing rule halts, and the app loads, shows correct verdicts and explains each step. |
| `results/` | All generated data, plots, trained models (`models.joblib`) and `summary.md`, which holds every number used in the report. |

---

## 3. How loop detection works

* A **configuration** is everything that decides the future: *(state, head position, whole tape)*.
* The machine is **deterministic**: the same configuration always leads to the same next step.
* So if a configuration ever **appears twice**, the machine will repeat the same cycle forever,
  and that is a **proof** that it never halts. We report `looping`.
* **How we notice a repeat:** we remember every configuration we have seen.
  * *Exact method* (used in the app): store `(state, head, frozenset(tape))` in a set. This is a
    definite proof, but slower.
  * *Fast method* (used to build the datasets): instead of copying the whole tape each step, keep
    a 64-bit fingerprint (a **Zobrist hash**) of the tape. When one cell changes, update the
    fingerprint in O(1) by XOR-ing out the old symbol and XOR-ing in the new one. Different
    tapes almost never share a fingerprint (chance ≈ 10⁻¹⁴ per run).
* **What it cannot catch:** a machine that never repeats, e.g. one that writes 1s to the right
  forever, or a binary counter. It never halts, but no configuration repeats, so we can only say
  `limit` ("don't know"). This is unavoidable, because the halting problem is undecidable.

---

## 4. Why the label is "halts within L" (and not "halts")

* "Does M halt?" is **undecidable**: no program can answer it for every machine. So we could
  never create correct labels for it, and no model could learn it.
* "Does M halt **within L steps**?" is **decidable**: simulate L steps and look. So every label in
  our dataset is **exactly correct** (we simulated every pair up to 1,000 steps).
* Label **0 means "has not halted within L steps"**, never "never halts". The machine might halt at
  step L + 1. The app and the report always word it that way.

---

## 5. Key results in 5 bullets

1. **Easy machines don't need ML.** On random machines, simply running 10 steps predicts "halts
   within 250" with **97.8%** accuracy (96% of halters stop within 10 steps). ML with only the rule
   table, without running anything, gets **81%**.
2. **Hard machines are where ML helps.** On machines still running after 10 steps, the baselines
   find **0** of the 301 test halters (F1 = 0). The Random Forest finds **84%** of them
   (F1 **0.527**, ROC-AUC **0.940**), and the Decision Tree **92%**.
3. **More simulation beats smarter models.** Raising the probe from 10 to 25 steps lifts F1 from
   0.53 to about **0.91**, because most hard halters stop between steps 10 and 25.
4. **Trees generalise, the neural net doesn't.** Trained on ≤ 6-state machines, the trees work
   equally well on unseen 7–10-state machines. The MLP's recall collapses from about 0.5 to 0.03.
5. **The neural net needed fixing for imbalance.** Unweighted, it found only 40% of the slow
   halters. Tuning its threshold on a validation split raised that to 79% (best F1 among the MLP variants).

---

## 6. Ten likely examiner questions

**Q1. Have you solved the halting problem?**
No, and nobody can: it is undecidable (Turing, 1936). We solve the *bounded* version ("within L
steps") exactly by simulation, and use ML only to *guess* it cheaply. Our class 0 means "not
halted within L", never "never halts".

**Q2. Why is the halting problem undecidable? (short proof)**
Suppose a program D could decide it. Build X: given a machine M, X asks D "does M halt on its own
description?" and then does the opposite (loops if D says halt, halts if D says loop). Now run X on
itself: if it halts it must loop, and if it loops it must halt. That is a contradiction, so D cannot exist.

**Q3. How can you be sure a "looping" machine never halts?**
The configuration (state, head, tape) repeated. A Turing machine is deterministic, so from the
same configuration it does exactly the same thing again, forever. It can never reach H.

**Q4. Why can't you detect every non-halting machine?**
Some machines never repeat a configuration, for example one that keeps writing new 1s to the
right. If we could always detect non-halting, we could decide the halting problem, which is
impossible. So we honestly report "limit reached".

**Q5. Why use ML at all if simulation is exact?**
Simulating costs time proportional to L. When L is huge, or there are many machines, a fast
prediction can *rank* which ones are worth a long run. ML is a triage tool. The final answer always
comes from simulation.

**Q6. What features did you use and which mattered most?**
Static features from the rule table (fraction of halting rules, distance to H in the state graph,
"run-away writer" rules…) and probe features from a 10-step run (did it halt, head span, net head
movement…). On random machines "did the probe already halt?" dominates. On hard machines it is
*can H be reached in the state graph* (`halting_rules_reachable`, `dist_to_halt`) together with
*is the head drifting away* (`probe_net_move`).

**Q7. Why is accuracy a bad metric for the hard machines?**
Only 8.4% of them halt. A model that always says "doesn't halt" is 91.6% accurate but useless. We
therefore report precision, recall, F1 and PR-AUC.

**Q8. How did you handle class imbalance?**
The trees use `class_weight="balanced"`, so errors on the rare class cost more. The MLP has no
class weights in scikit-learn, so it picks its decision threshold on a validation part of the
training data, choosing the threshold that gives the best F1 there. We compared this with oversampling and with
no fix (§3.3 of the report). We also found that early stopping made the MLP unstable on
imbalanced data, so we train for a fixed 60 epochs.

**Q9. How do you know the results are not just luck?**
Every experiment is repeated on 5 different random train/test splits, and we report mean ±
standard deviation (shown as error bars on the plots). Train and test sets never share a
`(machine, input)` pair, and all randomness is seeded, so the run is reproducible.

**Q10. How does prediction quality change with the step limit and with machine size?**
*Step limit:* a bigger L can only turn 0-labels into 1s. Once nothing new halts (after step ~87
for random machines and ~157 for hard ones), the labels and results stop changing. On hard machines F1
rises from 0.48 (L = 25) to 0.53 (L ≥ 50). *Machine size:* without running the machine, accuracy drops
as machines get bigger (0.85 at 2 states → 0.77 at 6). With the 10-step probe it stays around
0.97, and the tree models keep working on sizes they never saw in training.

**Bonus: What is a busy beaver and why include it?**
The n-state machine that runs the longest (or writes the most 1s) before halting, starting on a
blank tape. BB(3) halts after 14 steps, BB(4) after 107. They are *structured* slow halters, unlike
random machines, which makes them a good out-of-distribution test. The random-trained models miss them,
while the hard-trained trees catch BB(4).

---

## 7. Demo checklist (in the app)

0. Open the **Overview** tab: it shows where each requirement of the brief is demonstrated.
1. Pick **4-state busy beaver** → in the Simulator press **Next ▶** a few times: the "What happens
   next" box names the rule being applied and the transition table highlights it. Press **End ⏭**
   (step 107, halts) and watch the highlighted state reach H.
2. Pick **Ping-pong** → "Provably never halts": the configuration repeated.
3. Pick **Run-away right** → "Not halted within L": explain why this is not "never halts".
4. **Prediction tab** on BB(4): the per-limit table shows simulation saying "not within L" up to
   L = 100 and "halts" from L = 250. Compare the models' answers.
5. **Batch test** (hard, Random Forest, L = 250): recall far above the 10-step rule.
