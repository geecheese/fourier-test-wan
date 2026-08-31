#!/usr/bin/env python3
"""Read-only self-check for a Fourier test space on 1D Poisson.

Problem:
    -u'' = f on (0, pi),  u(0) = u(pi) = 0,
    u_exact(x) = sin(2x),  f(x) = 4 sin(2x).

No neural network is trained and no output file is written.
"""

import math

import numpy as np


PI = math.pi
MODE_COUNT = 32
ORDERS = (32, 64, 128, 256)


def gauss_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    points, weights = np.polynomial.legendre.leggauss(order)
    x = 0.5 * PI * (points + 1.0)
    w = 0.5 * PI * weights
    return x, w


def fourier_moments(
    x: np.ndarray,
    w: np.ndarray,
    trial_derivative: np.ndarray,
) -> np.ndarray:
    modes = np.arange(1, MODE_COUNT + 1, dtype=np.float64)[:, None]
    normalizer = math.sqrt(2.0 / PI)
    phi = normalizer * np.sin(modes * x[None, :])
    dphi = normalizer * modes * np.cos(modes * x[None, :])
    forcing = 4.0 * np.sin(2.0 * x)
    integrand = trial_derivative[None, :] * dphi - forcing[None, :] * phi
    return (integrand * w[None, :]).sum(axis=1)


def scores(moments: np.ndarray) -> tuple[float, float]:
    modes = np.arange(1, MODE_COUNT + 1, dtype=np.float64)
    raw = float(np.sum(moments**2))
    h_minus_one = float(np.sum(moments**2 / (1.0 + modes**2)))
    return raw, h_minus_one


def relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0e-30)


def main() -> None:
    expected_zero_raw = 8.0 * PI
    expected_zero_hminus1 = expected_zero_raw / 5.0
    results: dict[int, dict[str, float]] = {}

    print("=" * 88)
    print("FOURIER TEST-SPACE SELF-CHECK: 1D POISSON")
    print("=" * 88)
    print("Domain: (0, pi); exact solution: sin(2x); modes: 1,...,32")
    print("Definition: sum_k |R_k|^2/(1+k^2), with normalized sine modes")
    print("No training, parameter updates, or file writes.\n")

    for order in ORDERS:
        x, w = gauss_rule(order)

        exact_derivative = 2.0 * np.cos(2.0 * x)
        exact_moments = fourier_moments(x, w, exact_derivative)
        exact_raw, exact_hminus1 = scores(exact_moments)

        zero_moments = fourier_moments(x, w, np.zeros_like(x))
        zero_raw, zero_hminus1 = scores(zero_moments)

        results[order] = {
            "exact_hminus1": exact_hminus1,
            "zero_raw": zero_raw,
            "zero_hminus1": zero_hminus1,
        }

        largest = np.argsort(np.abs(zero_moments))[-3:][::-1] + 1
        print(f"QUADRATURE Q{order}")
        print(f"  exact: max_abs_moment={np.max(np.abs(exact_moments)):.3e}")
        print(f"         raw_score={exact_raw:.3e} Hminus1_score={exact_hminus1:.3e}")
        print(f"  zero : raw_score={zero_raw:.12e} Hminus1_score={zero_hminus1:.12e}")
        print(f"         three largest modes={largest.tolist()}\n")

    high = results[ORDERS[-1]]
    previous = results[ORDERS[-2]]
    exact_ok = high["exact_hminus1"] < 1.0e-24
    raw_ok = relative_difference(high["zero_raw"], expected_zero_raw) < 1.0e-12
    weighted_ok = (
        relative_difference(high["zero_hminus1"], expected_zero_hminus1) < 1.0e-12
    )
    refinement_ok = (
        relative_difference(high["zero_hminus1"], previous["zero_hminus1"])
        < 1.0e-12
    )

    print("=" * 88)
    print("ANALYTIC AND REFINEMENT CHECK")
    print("=" * 88)
    print(f"Expected zero-trial raw score       : 8*pi   = {expected_zero_raw:.12e}")
    print(
        "Expected zero-trial Hminus1 score   : 8*pi/5 = "
        f"{expected_zero_hminus1:.12e}"
    )
    print(
        "Q128/Q256 Hminus1 discrepancy       : "
        f"{relative_difference(high['zero_hminus1'], previous['zero_hminus1']):.3e}"
    )
    print(
        "Important: omitting 1/(1+k^2) produces the raw score, not the DFR H^{-1} score."
    )

    passed = exact_ok and raw_ok and weighted_ok and refinement_ok
    print(f"\nDECISION: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)
    print("The Fourier weak moments, spectral weights, and quadrature are consistent.")
    print("SELF-CHECK FINISHED.")


if __name__ == "__main__":
    main()