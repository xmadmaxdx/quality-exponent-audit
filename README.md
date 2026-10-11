# Is the Data-Quality Exponent Measurable?

**An identifiability audit of quality-aware scaling laws.**

[![status](https://img.shields.io/badge/status-NO%20RESULTS%20YET-criticalfire)](#1-current-status-read-this-first)
[![venue](https://img.shields.io/badge/target-TMLR-blueviolet)](#5-target-venue)
[![tests](https://img.shields.io/badge/tests-29%20passing-brightgreen)](tests/test_scaling.py)

---

## 1. Current status (read this first)

**There are no results yet. Nothing in this repository reports a finding.**

Every number that will eventually appear in the paper is produced by a script in
`scripts/` from stored run artifacts. No table is hand-typed, no figure is drawn
from a number typed by hand, and no placeholder value is presented as a result.
If you are looking for a headline number, it does not exist yet and the
repository says so rather than approximating it.

The pipeline has been verified end-to-end on synthetic data: a full artifact
sweep trains, stores bundles, fits all three fitters, bootstraps, and writes its
JSON. What has **not** happened is a single real CIFAR-100 run.

### Known open risk

A smoke-scale sweep (`--rates 0.1,0.5,0.9 --seeds 0,1`, synthetic data, 2 epochs)
produced variable-projection exponents that pinned to the optimiser bounds:

| operator | parabola | varpro | approach3 |
|---|---|---|---|
| `label_flip` | −2.434 | **+5.000 (bound)** | 1.538 |
| `label_noise` | 0.155 | −0.895 | 0.0001 |
| `pixel_gaussian` | 0.157 | **+3.024** | −0.0012 |
| `pixel_blur` | 0.149 | **−3.846 (bound)** | −3.866 |
| `pixel_mask` | 0.160 | 0.500 | −0.426 |

These are **not results** — they are a two-epoch synthetic smoke run with two
seeds, and the exponents should be meaningless noise. They are recorded because
they identify a real engineering risk: variable projection as implemented is
**unstable on short training schedules and low-quality data**, where the
optimiser walks the exponent to its bound instead of converging. Checkpoint C5
exists specifically to characterise this before any conclusion is drawn from it.
If varpro cannot be stabilised, the paper reports that as a finding about the
recommended alternative, not by quietly dropping the fitter.

---

## 2. The problem

Quality-aware scaling laws posit a **dimensionless data-quality parameter** `Q`
and fit a loss surface of the form

```
L(N, D, Q) ≈ A / N^a  +  B / (D^b · Q^γ)  +  E
```

where `γ` is the **quality exponent** — the number the field's central claims rest
on. The motivating question is quantitative: *how much does data quality matter,
and by how much does it matter more or less than data quantity?*

The form is current and load-bearing. Subramanyam, Chen & Grossman, *"Scaling Laws
Revited: Modeling the Role of Data Quality in Language Model Pretraining"*
(ICLR 2026, [arXiv:2510.03313](https://arxiv.org/abs/2510.03313)) introduce `Q`,
give two practical estimators for it, report fitted exponents of **γ ≈ 0.173**
(NMT) and **γ ≈ 0.401** (CLM), and validate the law out-of-sample — over 63 runs
on LLM pretraining.

**This repository audits whether that number is measurable at that sample size.**

### Why this is a gap, not a nitpick

Three things are true of the published work, and none of them is a criticism of
its quality:

1. **No confidence interval is attached to γ̂.** A point estimate from a finite
   grid is reported as though it were a constant of nature.
2. **The standard fitter is known to be biased on finite grids.** Chinchilla
   Approach 2 fits a parabola in `(log N, log D)` and reads off an exponent. The
   bias of exactly this estimator is documented (Czech et al.,
   arXiv:2603.22339). A biased estimator with no interval is not an estimate; it
   is a number.
3. **Between-seed variance is not part of the sample-size calculation.** Published
   grids are run-to-run, not seed-to-seed. The dispersion of a single training
   run at a fixed `(N, D, Q)` — `σ_run` — is the quantity every standard error
   downstream scales with, and it is not reported anywhere.

The audit's thesis is narrow, falsifiable, and negative in shape:

> **The field's central data-quality claim is statistically unidentified at its own
> reported sample size, and the fitted exponent is partly a property of the
> corruption operator used to define `Q`, not of the corpus.**

### The pre-committed falsifier

This matters more than the thesis, so it is stated before any data exists.

> **If `σ_run` is small and the parabola fitter is well-calibrated, then the
> published claims are fine and this paper's central correction is cosmetic.**

Checkpoint C2 measures `σ_run` and decides which paper gets written. Both
outcomes are publishable; only guessing is not. **C2 is the first experiment to
run and it gates everything downstream.**

---

## 3. What makes the exponent identifiable

Three fitters, all fed identical inputs, grids and seeds. Fairness here means
identical conditions — not identical code.

| fitter | method | role |
|---|---|---|
| `parabola` | Chinchilla Approach 2 — quadratic in `(log N, log D)` | **the fitter the literature uses.** Its finite-grid bias is the object of study. |
| `varpro` | variable projection — linear coefficients profiled out exactly, exponents searched | the recommended alternative in the bias literature |
| `approach3` | `L = A/N^a + B/(D^b·Q^γ) + E`, joint nonlinear least squares | the form Subramanyam et al. propose |

Two design properties are **enforced in code, not left to the caller**:

- **Nesting.** Every dataset-size point is a subset of the next larger one, drawn
  from one fixed permutation. A non-nested grid confounds dataset size with which
  examples happen to be present.
- **A held-out, disjoint, stratified eval split**, fixed across every run by a
  constant seed. Eval loss is the estimand, so the split must not move. It must
  also not overlap the fit pool — under label corruption, leakage would present
  clean original labels for examples whose *training* labels were deliberately
  damaged, manufacturing an apparent quality signal out of nothing.

Both are enforced in `src/scaling/grid.py` and `src/data/datasets.py`, and both
have regression tests.

---

## 4. The corruption-operator artifact

The sharpest experiment, and the cheapest decisive one.

Six operators damage a corpus. Each is parameterised by a single `rate` in
`[0, 1]`, and each is engineered to damage **the same fraction** of the corpus
while destroying a **different amount and kind of information**:

| operator | damages | mechanism |
|---|---|---|
| `label_flip` | target | structured, cyclic — looks learnable |
| `label_noise` | target | uniform at random |
| `pixel_gaussian` | input | additive, scaled to the image's own dynamic range |
| `pixel_blur` | input | low-pass |
| `pixel_mask` | input | occlusive |
| `duplicate` | neither | appends copies — grows the corpus with no new information |

`test_quality_operators_hold_rate_fixed` asserts the damaged fraction is within
`0.50 ± 0.05` for every operator. That constraint is what makes the test
possible: if the damaged fraction differed by operator, a fitted exponent could
be reading the *amount* of damage rather than its *mechanism*.

> **An exponent cannot be identified from a single quality level.** With quality
> fixed, `log Q` has no variance and no slope exists — the fitters raise rather
> than return a number. So every operator is characterised over the **same ladder
> of rates**, `γ̂` is fitted *within* each operator, and the question is whether
> those exponents agree. The comparison is at a matched severity *axis*, not at a
> single severity.

`duplicate` is reported separately and excluded from the headline spread: it
grows the corpus, so it perturbs the dataset-size axis rather than holding it
fixed. Pooling it would manufacture agreement or disagreement out of a confound.

**If `γ̂` differs across operators over the same rate ladder, the exponent
measures the instrument.**

---

## 5. Target venue

**TMLR — Transactions on Machine Learning Research.**

- **No article processing fee.** Verified live.
- Scope explicitly includes *"reproducibility studies of previously published
  results or claims"* and *"empirical evaluation of existing approaches"*, which
  is exactly this paper's shape.
- Has a **Reproducibility Certification** track.
- TMLR does **not** accept expanded versions of previously published work, so
  this manuscript shares no text with Projects 1 or 2.

TMLR is strict on claim–evidence correspondence and lenient on novelty framing.
That is the correct trade for a negative result with a rigorous audit trail.

---

## 6. Running it

### Colab (T4, free tier)

```
# Cell 1 — setup
!git clone https://github.com/xmadmaxdx/quality-exponent-audit
%cd quality-exponent-audit
!bash colab_setup.sh

# Cell 2 — mount Drive (results survive disconnects)
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/quality-exponent-audit-runs

# Cell 3 — data
!python -m scripts.download_data --sets cifar100

# Cell 4 — tests, offline and fast
!python -m pytest tests/test_scaling.py -q

# Cell 5 — C2: the load-bearing measurement
!python -m scripts.sigma_run --config configs/cifar100.yaml --seeds 0,1,2,3,4 \
    --download --out results/sigma_run.json

# Cell 6 — C7: the artifact test
!python -m scripts.artifact_test --config configs/cifar100.yaml \
    --rates 0.1,0.3,0.5,0.7,0.9 --seeds 0,1,2,3,4 --download \
    --out results/artifact.json

# Cell 7 — tables, regenerated from stored artifacts
!python -m scripts.make_tables --runs outputs/runs
```

### Local

```powershell
pip install -r requirements.txt
python -m pytest tests/test_scaling.py -q
python -m scripts.artifact_test --config configs/smoke.yaml   # synthetic, ~3 min
```

> **On Windows:** `timeout` is the `TIMEOUT.EXE` built-in, not GNU coreutils.
> Use `timeout /t 900`, or omit it — a bare `timeout 900` fails with
> *"Invalid syntax. Default option is not allowed more than '1' time(s)"*, which
> looks like a code error and is not.

---

## 7. Checkpoints

**This is the evidence chain.** Each checkpoint produces a file, gates the next,
and maps to a specific claim in the paper. A paper result is admissible only if
the checkpoint that produces it passed.

Run them in order. C2 is the gate that decides which paper is being written.

| # | checkpoint | status | produces | gates | command |
|---|---|---|---|---|---|
| **C0** | environment & determinism | done | stored per run | everything | automatic |
| **C1** | fitter recovery | done | `pytest -k recovers` | C4, C5 | `pytest -k recovers` |
| **C2** | **σ_run** — *decision gate* | done | `results/sigma_run.json` | **all** | `scripts.sigma_run` |
| **C3** | closed-form vs bootstrap SE | *planned* | `results/se_calibration.json` | C6 | `scripts.se_calibration` |
| **C4** | interval coverage | *planned* | `results/coverage.json` | C5, headline | `scripts.coverage` |
| **C5** | fitter bias comparison — *carries the varpro stability risk, §1* | *planned* | `results/fitter_bias.json` | C6, Table 2 | `scripts.fitter_bias` |
| **C6** | runs-needed curve | *planned* | `results/runs_needed.json` | C8, **Figure 2** | `scripts.runs_needed` |
| **C7** | corruption-operator artifact | done | `results/artifact.json` | C8, **Figure 1** | `scripts.artifact_test` |
| **C8** | applied audit of published numbers | *planned* | `results/published_audit.json` | abstract, Table 1 | `scripts.apply_to_published` |
| **C9** | robustness | *planned* | `results/robustness.json` | reviewer defence | `scripts.robustness` |

**done** — the code exists and has been executed at least once.
**planned** — specified below but **not yet written**. The command names are the
intended interface; those scripts do not exist yet and no result is attributed
to them.

**C1, C2 and C7 are the only checkpoints currently executable.** C2 and C7 have
run on synthetic data only, never on CIFAR-100. The estimator machinery they
depend on (`src/scaling/inference.py`) is complete and unit-tested; C3–C6 and
C8–C9 are scripts that consume it rather than new science.

### The gates in detail

**C0 — Environment manifest.** Every run bundle records Python, platform, Torch,
CUDA, cuDNN and device name. cuDNN autotuning and non-deterministic kernels are
pinned explicitly (`set_seed` sets `cudnn.deterministic=True`,
`cudnn.benchmark=False`) because they are separate sources of the very variance
this project measures, and leaving them to defaults would make `σ_run`
irreproducible for reasons that have nothing to do with the science.

**C1 — Fitter recovery.** On synthetic data generated *from* the Approach-3 law
with `γ = 0.40` known by construction, `approach3` recovers `0.40` to within
`0.02` and `varpro` to within `0.05`. **If a fitter cannot recover a truth it was
built from, no claim about identifiability built on it means anything.** This gate
is why the smoke numbers in §1 are interpretable as noise rather than as findings.

**C2 — σ_run.** Train at one fixed `(N, D, Q)` cell across ≥ 5 seeds, on clean
data — corruption is deliberately excluded here, because `σ_run` is the
dispersion of the *training process*, and introducing the factor under study
would confound the measurement with it. Emits `σ_run`, the standard error of
`γ̂` at each candidate grid size, the minimum detectable gap, and the number of
runs needed to resolve a `0.10` gap. **Decision point, per §2.**

**C3 — SE calibration.** The closed form
`SE(γ̂) = √(2 σ² / (g · sd²(log D)))` — `√2` because the exponent is a difference
of two slopes — is checked against the bootstrap SE. If they disagree, the
closed form is wrong and every derived sample-size claim inherits the error.

**C4 — Coverage.** The decisive test. Run the whole pipeline on synthetic data
with a known `γ`, across many simulated grids, and ask how often a nominal 95%
interval actually covers. An estimator can have a small bootstrap SE and still be
badly biased, in which case the interval is *narrow and wrong* — confidently
misleading. Coverage below nominal is the finding; a narrow interval around a
biased centre is exactly what the published `γ` estimates look like.

**C5 — Fitter bias.** Recovery error of each of the three fitters against known
truth, as a function of grid size and `σ_run`. Directly quantifies how much of
Approach 2's bias contaminates a quality exponent. Includes the varpro
stability check flagged in §1.

**C6 — Runs-needed curve.** Inverts C3 to give, for a target `γ` gap, the number
of runs required as a function of `σ_run` and grid size. **This is the number the
literature has never reported.** It converts the argument from rhetoric into
arithmetic: not "more error bars would be nice" but "*N* runs are needed, and
published grids used fewer."

**C7 — Artifact test.** §4. Reports `γ̂` per operator over a shared rate ladder,
each with a bootstrap SE, plus the spread across operators measured against that
noise. **The paper's best figure.**

**C8 — Applied audit.** Refit the published `γ̂` values under the corrected
estimator with attached intervals, and state plainly how many of the literature's
qualitative conclusions survive. A negative result is only credible if it is
applied to the incumbent's own numbers.

**C9 — Robustness.** Repeat across dataset, model width, optimiser and schedule.
Any single-regime finding is not a finding.

---

## 8. Reproducibility contract

The code that runs but stores nothing cannot be defended in review.

Every run writes a self-contained bundle under
`outputs/runs/<study>/N<params>_D<examples>_Q<quality>_<operator>_s<seed>/`:

| artifact | contents |
|---|---|
| `metrics.json` | fully resolved config, grid point, environment manifest, all metrics |
| `train_log.jsonl` | per-epoch train/eval loss, accuracy, wall time |
| `preds.npy` | raw per-example softmax predictions |
| `labels.npy` | raw eval labels |
| `last.pt` | model, optimizer, scheduler, epoch, **and full RNG state** |
| `best.pt` | best checkpoint by eval loss |
| `DONE` | completion sentinel — partial runs are skipped, never silently averaged |

- **Resumable.** A Colab disconnect costs at most one run, not a grid. A failed
  resume is logged loudly and restarts from scratch rather than discarding
  completed epochs invisibly.
- **Re-derivable.** Raw predictions are stored, so any metric can be recomputed
  after the fact without retraining.
- **Script-generated tables.** `scripts/make_tables.py` builds every table from
  stored artifacts and refuses to fit on fewer than three cells.
- **No fabricated results.** `results/` is gitignored. Nothing in the repository
  is a number that was not produced by a script.

---

## 9. Repository layout

```
configs/
  smoke.yaml              synthetic, offline, ~3 min — the end-to-end check
  cifar100.yaml           the real study
src/
  constants.py            every named constant, including EVAL_SPLIT_SEED
  config.py               typed YAML loading, fail-closed validation
  data/
    datasets.py           split_indices (disjointness-verifiable), nested subsets
    quality.py            six corruption operators, rate-fixed by construction
  engine/trainer.py       resumable training, complete artifact storage
  models/factory.py       CIFAR ResNet sized by `width` — the N axis
  scaling/
    fitters.py            parabola, varpro, approach3 — identical inputs
    grid.py               nesting and axis-crossing enforced
    inference.py          σ_run, SE(γ̂), minimum detectable gap, runs needed
scripts/
  sigma_run.py            C2 — the load-bearing measurement
  artifact_test.py        C7 — the headline experiment
  make_tables.py          every table, from artifacts
  download_data.py        Colab entry point
tests/test_scaling.py     29 tests, each written against a specific defect
```

---

## 10. Engineering notes

Three defects found and fixed during construction, recorded because the pattern
recurs and the tests now guard it:

- **Train/eval leakage.** The eval set was initially a prefix slice of the corpus
  being trained on. Under label corruption this is not merely optimistic — it
  presents clean labels for deliberately damaged examples and manufactures a
  quality signal from nothing. Fixed by `split_indices`, which returns indices
  so disjointness is a checkable claim, and asserts non-overlap at runtime.
- **A fitter that cannot fit a known truth.** `varpro` recovered `0.083` for a
  truth of `0.40`, because it omitted the irreducible-error term and absorbed a
  constant into `γ`, biasing it toward zero. The offset is now in the profiled
  basis.
- **Non-ASCII identifiers inside optimiser callables.** `scipy.optimize.least_squares`
  compiles its objective through Cython, which rejects Greek identifiers with the
  opaque *"Default option is not allowed more than '1' time(s)"*. A lambda
  capturing a Python-level name happens to dodge it; nothing here relies on that
  accident, and a source-level test now enforces ASCII in optimiser callables.

---

## 11. Citation

Subramanyam, A., Chen, Y., & Grossman, R. L. (2026). *Scaling Laws Revisited:
Modeling the Role of Data Quality in Language Model Pretraining.* ICLR 2026.
[arXiv:2510.03313](https://arxiv.org/abs/2510.03313).
