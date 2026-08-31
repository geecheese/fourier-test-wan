#!/usr/bin/env python3
"""Read-only Fourier weak-form self-check for a 1D Dirac point source.

Problem:
    -u'' = delta_{pi/2} on (0, pi),  u(0)=u(pi)=0.

The exact solution is the Green's-function tent profile.  It is in H^1_0 but
not H^2.  Quadrature is split at the kink x=pi/2.
"""

import math

import numpy as np


PI = math.pi
SOURCE = 0.5 * PI
MODES = 64
ORDERS = (32, 64, 128, 256)


def interval_rule(order: int, left: float, right: float):
    points, weights = np.polynomial.legendre.leggauss(order)
    x = 0.5 * (right - left) * points + 0.5 * (right + left)
    w = 0.5 * (right - left) * weights
    return x, w


def split_rule(order: int):
    x_left, w_left = interval_rule(order, 0.0, SOURCE)
    x_right, w_right = interval_rule(order, SOURCE, PI)
    return (
        np.concatenate((x_left, x_right)),
        np.concatenate((w_left, w_right)),
    )


def exact_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x < SOURCE, (PI - SOURCE) / PI, -SOURCE / PI)


def residual_moments(order: int, include_source: bool) -> np.ndarray:
    x, w = split_rule(order)
    k = np.arange(1, MODES + 1, dtype=np.float64)[:, None]
    normalizer = math.sqrt(2.0 / PI)
    dphi = normalizer * k * np.cos(k * x[None, :])
    volume = (dphi * exact_derivative(x)[None, :] * w[None, :]).sum(axis=1)
    point_load = normalizer * np.sin(k[:, 0] * SOURCE)
    return volume - point_load if include_source else volume


def weighted_score(moments: np.ndarray) -> float:
    k = np.arange(1, MODES + 1, dtype=np.float64)
    return float(np.sum(moments**2 / (1.0 + k**2)))


def relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0e-300)


def main() -> None:
    records = {}
    print("=" * 96)
    print("FOURIER TEST-SPACE SELF-CHECK: 1D DIRAC POINT SOURCE")
    print("=" * 96)
    print("Problem: -u''=delta_{pi/2} on (0,pi), with homogeneous Dirichlet boundary")
    print("Exact solution: Green's-function tent profile in H1_0 but not H2")
    print("Quadrature is split at x=pi/2; modes=1,...,64")
    print("No model training, parameter updates, or file writes.\n")

    for order in ORDERS:
        correct = residual_moments(order, include_source=True)
        omitted = residual_moments(order, include_source=False)
        correct_score = weighted_score(correct)
        omitted_score = weighted_score(omitted)
        records[order] = (correct_score, omitted_score)
        print(f"SPLIT QUADRATURE Q{order}+Q{order}")
        print(
            f"  correct point-source form: score={correct_score:.12e} "
            f"max_abs_moment={np.max(np.abs(correct)):.3e}"
        )
        print(
            f"  source term omitted      : score={omitted_score:.12e} "
            f"max_abs_moment={np.max(np.abs(omitted)):.3e}\n"
        )

    correct_high, omitted_high = records[ORDERS[-1]]
    correct_previous, omitted_previous = records[ORDERS[-2]]
    correct_gap = relative_difference(correct_high, correct_previous)
    omitted_gap = relative_difference(omitted_high, omitted_previous)

    # The correct score is roundoff-sized, so use an absolute convergence test.
    passed = (
        correct_high < 1.0e-22
        and omitted_high > 1.0e-3
        and abs(correct_high - correct_previous) < 1.0e-22
        and omitted_gap < 1.0e-12
    )

    print("=" * 96)
    print("SELF-CHECK DECISION")
    print("=" * 96)
    print(f"Correct Q128/Q256 relative indicator : {correct_gap:.3e}")
    print(f"Correct Q128/Q256 absolute difference: {abs(correct_high-correct_previous):.3e}")
    print(f"Omitted Q128/Q256 discrepancy        : {omitted_gap:.3e}")
    print(f"Correct refined score                : {correct_high:.12e}")
    print(f"Omitted refined score                : {omitted_high:.12e}")
    print(f"\nDECISION: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)
    print("The Fourier weak form resolves the Dirac source and detects its omission.")
    print("POINT-SOURCE SELF-CHECK FINISHED.")


if __name__ == "__main__":
    main()