"""Quality manipulation operators.

The design constraint that makes the artifact test possible: every operator is
parameterised by a single `rate` in [0, 1] meaning "fraction of the corpus
damaged". Holding `rate` fixed across operators holds the *amount* of corruption
constant while changing the *mechanism*.

That is the whole point. If a fitted quality exponent moves when only the
mechanism changes at fixed rate, the exponent is a property of the corruption
operator rather than of the corpus, and any law built on it is measuring the
damper.
"""
from __future__ import annotations

import numpy as np

from src.constants import CORRUPTION_OPERATORS

# Blur kernel size at maximum severity, in pixels. Small images keep this modest
# so that a blurred image is still learnable rather than uniformly grey.
BLUR_KERNEL = 3

# Operators partitioned by what they damage. Keeping these separate is what
# guarantees each operator applies exactly one corruption.
LABEL_OPERATORS = frozenset({"label_flip", "label_noise"})
PIXEL_OPERATORS = frozenset({"pixel_gaussian", "pixel_blur", "pixel_mask"})

_UNKNOWN_OPERATOR = set(CORRUPTION_OPERATORS) - LABEL_OPERATORS - PIXEL_OPERATORS - {"duplicate"}
if _UNKNOWN_OPERATOR:  # pragma: no cover - guards against an unclassified operator
    raise RuntimeError(f"operators not classified: {sorted(_UNKNOWN_OPERATOR)}")


def operator_names() -> tuple[str, ...]:
    return CORRUPTION_OPERATORS


def corrupt_labels(labels: np.ndarray, rate: float, rng: np.random.Generator,
                   num_classes: int, operator: str) -> np.ndarray:
    """Apply a target-corrupting operator. Returns a new label array.

    `label_flip` sends a damaged example to the next class cyclically, which is a
    structured, learnable-looking error. `label_noise` sends it uniformly at
    random. Both damage `rate` of the corpus but destroy different amounts of
    information, so a fitted exponent that treats them as the same quality
    level is measuring the operator.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    out = labels.copy()
    n = labels.size
    if n == 0 or rate == 0.0:
        return out
    damaged = rng.random(n) < rate
    idx = np.flatnonzero(damaged)
    if idx.size == 0:
        return out
    if operator == "label_flip":
        out[idx] = (labels[idx] + 1) % num_classes
    elif operator == "label_noise":
        out[idx] = rng.integers(0, num_classes, size=idx.size)
    else:
        raise ValueError(f"{operator} does not corrupt labels")
    return out


def corrupt_pixels(images: np.ndarray, rate: float, rng: np.random.Generator,
                   operator: str, image_size: int) -> np.ndarray:
    """Apply an input-corrupting operator. Returns a new image array.

    `pixel_gaussian` is additive noise, `pixel_blur` is low-pass filtering, and
    `pixel_mask` is occlusive. All three damage `rate` of the images but carry
    different signal, which is the comparison the artifact test needs.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    out = images.copy()
    n = out.shape[0]
    if n == 0 or rate == 0.0:
        return out
    damaged = rng.random(n) < rate
    idx = np.flatnonzero(damaged)
    if idx.size == 0:
        return out
    if operator == "pixel_gaussian":
        # Scale noise to the image's own dynamic range so severity is comparable
        # across datasets rather than a fixed absolute amount.
        scale = out[idx].std()
        out[idx] = out[idx] + rng.normal(0.0, max(scale, 1e-3), size=out[idx].shape)
    elif operator == "pixel_blur":
        from PIL import Image, ImageFilter

        for i in idx:
            pil = Image.fromarray(out[i]).filter(ImageFilter.BoxBlur(BLUR_KERNEL))
            out[i] = np.asarray(pil.convert("RGB" if out.shape[-1] == 3 else "L"))
    elif operator == "pixel_mask":
        side = max(1, image_size // 4)
        for i in idx:
            y = rng.integers(0, image_size - side + 1)
            x = rng.integers(0, image_size - side + 1)
            out[i, y:y + side, x:x + side] = 0
    else:
        raise ValueError(f"{operator} does not corrupt pixels")
    return out


def duplicate_data(images: np.ndarray, labels: np.ndarray, rate: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Append duplicated examples so the corpus grows without new information.

    This is the only operator that leaves both inputs and labels intact. It is
    the cleanest demonstration of the confound: if `rate` means "quality" then
    duplication is pure information loss with no corruption at all.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    if images.shape[0] == 0 or rate == 0.0:
        return images, labels
    # Duplicate `rate` of the corpus, so the effective unique fraction is 1-rate.
    n_extra = int(round(images.shape[0] * rate))
    if n_extra == 0:
        return images, labels
    src = rng.integers(0, images.shape[0], size=n_extra)
    return (np.concatenate([images, images[src]], axis=0),
            np.concatenate([labels, labels[src]], axis=0))


def apply_quality(images: np.ndarray, labels: np.ndarray, operator: str, rate: float,
                  seed: int, num_classes: int, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply one operator at a given rate. Returns (images, labels).

    Seeded so that the corruption draw is reproducible and independent of the
    training seed, which keeps quality manipulation a controlled factor rather
    than an uncontrolled one.
    """
    if operator not in CORRUPTION_OPERATORS:
        raise ValueError(f"unknown operator: {operator}; allowed {CORRUPTION_OPERATORS}")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    rng = np.random.default_rng(seed)
    if operator == "duplicate":
        return duplicate_data(images, labels, rate, rng)
    # Label operators damage targets only; pixel operators damage inputs only.
    # Applying both would corrupt twice and break the fixed-rate comparison that
    # the artifact test depends on.
    if operator in LABEL_OPERATORS:
        labels = corrupt_labels(labels, rate, rng, num_classes, operator)
    if operator in PIXEL_OPERATORS:
        images = corrupt_pixels(images, rate, rng, operator, image_size)
    return images, labels