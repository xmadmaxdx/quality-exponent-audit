"""Shared named constants. No magic numbers anywhere else in the code."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Estimator settings
# ---------------------------------------------------------------------------
# Bootstrap replicates used for the sampling distribution of the quality
# exponent. Small on purpose: this project is about estimator noise, so the
# bootstrap must not be the dominant source of randomness.
BOOTSTRAP_REPLICATES = 400

# Standard log-log regression slope factor used in the standard error of an
# exponent estimated over g grid points. Chosen in run 1 from measurement.
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80

# Number of standard errors defining a minimum detectable gap.
Z_ONE_SIDED = 1.6448536269514722
Z_TWO_SIDED = 1.959963984540054

# ---------------------------------------------------------------------------
# Training protocol
# ---------------------------------------------------------------------------
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 5e-4
SEED = 0

# Minimum seeds per grid cell. The entire project is a variance study, so a
# single seed anywhere is a defect rather than a shortcut.
MIN_SEEDS_PER_CELL = 5

# Fixed seed for the held-out eval split. Constant across every run and every
# study, because eval loss is the estimand being fitted: a split that moved
# between runs would put split variance into the fitted exponent.
EVAL_SPLIT_SEED = 12345

# ---------------------------------------------------------------------------
# Quality manipulation
# ---------------------------------------------------------------------------
# Quality is a probability in [0, 1]: the fraction of training examples left
# intact. Q = 1.0 is clean data.
QUALITY_MIN = 0.0
QUALITY_MAX = 1.0

# Severities used for corruption operators. Held fixed across operators so the
# artifact test holds corruption rate constant while varying its mechanism.
DEFAULT_SEVERITIES = (1, 2, 3, 4, 5)

# Operators deliberately chosen to have the same rate but different mechanisms.
# The paper's central artifact claim is that the fitted exponent tracks this
# choice rather than the underlying corpus.
CORRUPTION_OPERATORS = (
    "label_flip",       # corrupts the target, not the input
    "label_noise",      # corrupts the target to a random class
    "pixel_gaussian",   # corrupts the input, additive
    "pixel_blur",       # corrupts the input, low-pass
    "pixel_mask",       # corrupts the input, occlusive
    "duplicate",        # leaves labels intact, removes information
)

# ---------------------------------------------------------------------------
# Grid sizes for the identifiability study
# ---------------------------------------------------------------------------
GRID_SIZES = (15, 30, 60, 120)

# Nested subsets are what make the exponent identifiable at all; a non-nested
# grid confounds dataset size with quality.
REQUIRE_NESTED = True

# ---------------------------------------------------------------------------
# Numeric guards
# ---------------------------------------------------------------------------
LOG_EPS = 1e-12
MIN_LOGIT = 1e-9
EPS = 1e-6