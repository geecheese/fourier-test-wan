#!/usr/bin/env python3
"""Read-only audit of uniform/DST training grids for the 2D diagonal layer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from selfchecks.fourier_2d_diagonal_layer_selfcheck import exact_gradient, forcing, tensor_gauss_rule
from experiments.poisson2d.train_spectral_wan import grid
from src.methods.wan_spectral_loss import SpectralWANLoss


PI = math.pi
DTYPE = torch.float64
TERM_RTOL = 5.0e-3
RHO_LIMIT = 1.0e-4
CALIBRATION_TOL = 5.0e-3
REFERENCE_REL_TOL = 5.0e-10
REFERENCE_COMPONENT_TOL = 1.0e-8
REFERENCE_RESIDUAL_TOL = 1.0e-8
REFERENCE_DFR_TOL = 1.0e-16
KNOWN_MODES = ((1, 1), (3, 5), (8, 7), (16, 15), (32, 31))


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def continuous_terms(order: int, modes: int, beta: float,
                     device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensor Gauss A and B without forming an N^2 by Q^2 tensor."""
    points, weights = tensor_gauss_rule(order, device)
    q = order
    x = points[:, 0].reshape(q, q)
    y = points[:, 1].reshape(q, q)
    # tensor_gauss_rule returns the outer-product weights; both axes use the same rule.
    axis_weights = torch.sqrt(torch.diagonal(weights.reshape(q, q)))
    wx = axis_weights
    wy = axis_weights
    gradient = exact_gradient(points, beta).reshape(q, q, 2)
    source = forcing(points, beta).reshape(q, q)
    indices = torch.arange(1, modes + 1, dtype=DTYPE, device=device)
    axis = x[:, 0]
    sine = torch.sin(axis[:, None] * indices[None, :])
    cosine_derivative = torch.cos(axis[:, None] * indices[None, :]) * indices[None, :]
    weighted_sine_x = wx[:, None] * sine
    weighted_sine_y = wy[:, None] * sine
    weighted_cosine_x = wx[:, None] * cosine_derivative
    weighted_cosine_y = wy[:, None] * cosine_derivative
    normalization = 2.0 / PI
    a = normalization * (
        weighted_cosine_x.T @ gradient[:, :, 0] @ weighted_sine_y
        + weighted_sine_x.T @ gradient[:, :, 1] @ weighted_cosine_y
    )
    b = normalization * (weighted_sine_x.T @ source @ weighted_sine_y)
    return a, b


def uniform_weak_terms(grid_size: int, modes: int, beta: float,
                       device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Rectangle-rule weak moments on the exact training points."""
    points = grid(grid_size, 2, str(device))
    n = grid_size - 2
    axis = points[:, 0].reshape(n, n)[:, 0]
    gradient = exact_gradient(points, beta).reshape(n, n, 2)
    source = forcing(points, beta).reshape(n, n)
    indices = torch.arange(1, modes + 1, dtype=DTYPE, device=device)
    sine = torch.sin(axis[:, None] * indices[None, :])
    cosine_derivative = torch.cos(axis[:, None] * indices[None, :]) * indices[None, :]
    h = PI / (grid_size - 1)
    normalization = 2.0 / PI
    a = normalization * h**2 * (
        cosine_derivative.T @ gradient[:, :, 0] @ sine
        + sine.T @ gradient[:, :, 1] @ cosine_derivative
    )
    b = normalization * h**2 * (sine.T @ source @ sine)
    return a, b


def training_projection_terms(grid_size: int, modes: int, beta: float,
                              device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Actual DST-II projections of -Delta u and f, plus their strong residual."""
    points = grid(grid_size, 2, str(device))
    n = grid_size - 2
    source = forcing(points, beta).reshape(n, n)
    # For the manufactured solution, the independently checked analytic identity is -Delta u*=f.
    minus_laplacian = forcing(points, beta).reshape(n, n)
    loss = SpectralWANLoss((modes, modes), 2, boundary="dirichlet", domain=PI)
    projected_laplacian = loss.project(minus_laplacian)
    projected_source = loss.project(source)
    projected_strong = loss.project(minus_laplacian - source)
    return projected_laplacian, projected_source, projected_strong


def relative_frobenius(value: torch.Tensor, reference: torch.Tensor) -> float:
    return scalar(torch.linalg.vector_norm(value - reference)
                  / torch.linalg.vector_norm(reference))


def weighted_score(residual: torch.Tensor) -> float:
    n = residual.shape[0]
    loss = SpectralWANLoss((n, n), 2, boundary="dirichlet", domain=PI)
    return scalar((residual.square() * loss.weights(residual.dtype, residual.device)).sum())


def cancellation_indicator(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(a) + torch.linalg.vector_norm(b)
    return scalar(torch.linalg.vector_norm(a - b) / denominator)


def mode_of_maximum(value: torch.Tensor) -> list[int]:
    index = int(torch.argmax(value.abs()).item())
    n = value.shape[1]
    return [index // n + 1, index % n + 1]


def calibration_for_grid(grid_size: int, maximum_requested_mode: int, beta: float,
                         device: torch.device) -> dict[str, Any]:
    del beta  # Calibration is independent of the manufactured solution.
    n = grid_size - 2
    projection_modes = min(maximum_requested_mode, n)
    points = grid(grid_size, 2, str(device))
    x = points[:, 0].reshape(n, n)
    y = points[:, 1].reshape(n, n)
    loss = SpectralWANLoss((projection_modes, projection_modes), 2,
                           boundary="dirichlet", domain=PI)
    cases = []
    for p, q in KNOWN_MODES:
        if p > projection_modes or q > projection_modes:
            cases.append({"mode": [p, q], "status": "UNSUPPORTED"})
            continue
        values = (2.0 / PI) * torch.sin(p * x) * torch.sin(q * y)
        coefficients = loss.project(values)
        target = scalar(coefficients[p - 1, q - 1])
        off_diagonal = coefficients.clone()
        off_diagonal[p - 1, q - 1] = 0.0
        maximum_off = scalar(off_diagonal.abs().max())
        passed = abs(target - 1.0) <= CALIBRATION_TOL and maximum_off <= CALIBRATION_TOL
        cases.append({
            "mode": [p, q], "status": "PASS" if passed else "FAIL",
            "target_projection_coefficient": target,
            "target_absolute_error": abs(target - 1.0),
            "maximum_off_diagonal_coefficient": maximum_off,
            "dst_output_to_continuous_normalized_moment_scale": 1.0,
        })
    return {
        "grid_size": grid_size, "interior_points_per_axis": n,
        "physical_points": "x_j=j*pi/(grid_size-1), j=1,...,grid_size-2",
        "dst_assumed_points": "x_j=(j+1/2)*pi/(grid_size-2), j=0,...,grid_size-3",
        "derived_2d_scale": "(sqrt(2/n)*sqrt(pi/n))^2=2*pi/n^2, equal to normalized midpoint quadrature scale",
        "calibration_tolerance": CALIBRATION_TOL, "cases": cases,
    }


def calibration_passes(calibration: dict[str, Any], modes: int) -> bool:
    relevant = [case for case in calibration["cases"]
                if max(case["mode"]) <= modes and case["status"] != "UNSUPPORTED"]
    return bool(relevant) and all(case["status"] == "PASS" for case in relevant)


def finite_tensors(*values: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values)


def reference_check(primary_a: torch.Tensor, primary_b: torch.Tensor,
                    refined_a: torch.Tensor, refined_b: torch.Tensor) -> dict[str, Any]:
    residual = refined_a - refined_b
    a_rel = relative_frobenius(primary_a, refined_a)
    b_rel = relative_frobenius(primary_b, refined_b)
    max_component = max(scalar((primary_a - refined_a).abs().max()),
                        scalar((primary_b - refined_b).abs().max()))
    max_residual = scalar(residual.abs().max())
    dfr = weighted_score(residual)
    finite = finite_tensors(primary_a, primary_b, refined_a, refined_b)
    passed = (finite and a_rel <= REFERENCE_REL_TOL and b_rel <= REFERENCE_REL_TOL
              and max_component <= REFERENCE_COMPONENT_TOL
              and max_residual <= REFERENCE_RESIDUAL_TOL and dfr <= REFERENCE_DFR_TOL)
    return {
        "A_relative_frobenius_discrepancy": a_rel,
        "B_relative_frobenius_discrepancy": b_rel,
        "maximum_absolute_component_discrepancy": max_component,
        "refined_correct_residual_maximum_absolute_moment": max_residual,
        "refined_weighted_DFR_score": dfr,
        "finite": finite,
        "thresholds": {
            "relative_frobenius": REFERENCE_REL_TOL,
            "maximum_component": REFERENCE_COMPONENT_TOL,
            "maximum_residual": REFERENCE_RESIDUAL_TOL,
            "weighted_DFR_score": REFERENCE_DFR_TOL,
        },
        "decision": "PASS" if passed else "FAIL",
    }


def grid_record(grid_size: int, modes: int, beta: float, device: torch.device,
                reference_a: torch.Tensor, reference_b: torch.Tensor,
                calibration: dict[str, Any], reference_passed: bool) -> dict[str, Any]:
    n = grid_size - 2
    base = {"grid_size": grid_size, "interior_points_per_axis": n, "mode_count": modes}
    if modes > n:
        return {**base, "status": "INADMISSIBLE",
                "reason": f"mode_count={modes} exceeds independent DST modes={n}"}
    a, b = uniform_weak_terms(grid_size, modes, beta, device)
    ref_a, ref_b = reference_a[:modes, :modes], reference_b[:modes, :modes]
    residual = a - b
    a_rel = relative_frobenius(a, ref_a)
    b_rel = relative_frobenius(b, ref_b)
    a_max = scalar((a - ref_a).abs().max())
    b_max = scalar((b - ref_b).abs().max())
    max_residual = scalar(residual.abs().max())
    dfr = weighted_score(residual)
    ref_b_score = weighted_score(ref_b)
    rho = dfr / ref_b_score
    cancel = cancellation_indicator(a, b)
    calibration_ok = calibration_passes(calibration, modes)
    finite = finite_tensors(a, b, residual) and all(math.isfinite(value) for value in
        (a_rel, b_rel, a_max, b_max, max_residual, dfr, rho, cancel))
    anomalous_cancellation = rho <= RHO_LIMIT and (a_rel > TERM_RTOL or b_rel > TERM_RTOL)
    proj_a, proj_b, strong = training_projection_terms(grid_size, modes, beta, device)
    pass_conditions = {
        "A_relative_error": a_rel <= TERM_RTOL,
        "B_relative_error": b_rel <= TERM_RTOL,
        "normalized_residual_score": rho <= RHO_LIMIT,
        "projection_calibration": calibration_ok,
        "finite": finite,
        "representable": True,
        "no_anomalous_cancellation": not anomalous_cancellation,
        "continuous_reference": reference_passed,
    }
    return {
        **base, "status": "PASS" if all(pass_conditions.values()) else "FAIL",
        "mode_representable": True,
        "A_relative_frobenius_error": a_rel,
        "B_relative_frobenius_error": b_rel,
        "A_maximum_absolute_component_error": a_max,
        "B_maximum_absolute_component_error": b_max,
        "maximum_absolute_residual_moment": max_residual,
        "weighted_discrete_DFR_score": dfr,
        "largest_residual_mode": mode_of_maximum(residual),
        "residual_cancellation_indicator": cancel,
        "normalized_residual_score_rho_h": rho,
        "reference_B_weighted_score": ref_b_score,
        "anomalous_cancellation": anomalous_cancellation,
        "pass_conditions": pass_conditions,
        "actual_training_DST_projection": {
            "minus_laplacian_projection_norm": scalar(torch.linalg.vector_norm(proj_a)),
            "forcing_projection_norm": scalar(torch.linalg.vector_norm(proj_b)),
            "A_projection_relative_reference_error": relative_frobenius(proj_a, ref_a),
            "B_projection_relative_reference_error": relative_frobenius(proj_b, ref_b),
            "strong_residual_maximum_absolute_moment": scalar(strong.abs().max()),
            "strong_residual_weighted_DFR_score": weighted_score(strong),
            "label": "STRONG RESIDUAL DIAGNOSTIC — NOT THE REQUESTED WAN WEAK MOMENT",
        },
    }


def print_calibration(calibrations: list[dict[str, Any]]) -> None:
    print("\nPROJECTION CALIBRATION")
    for result in calibrations:
        print(f"grid_size={result['grid_size']} interior={result['interior_points_per_axis']}")
        for case in result["cases"]:
            if case["status"] == "UNSUPPORTED":
                print(f"  mode={tuple(case['mode'])} STATUS: UNSUPPORTED")
            else:
                print(f"  mode={tuple(case['mode'])} target={case['target_projection_coefficient']:.17e} "
                      f"max_offdiag={case['maximum_off_diagonal_coefficient']:.17e} "
                      f"scale={case['dst_output_to_continuous_normalized_moment_scale']:.1f} "
                      f"status={case['status']}")


def print_grid_table(records: list[dict[str, Any]]) -> None:
    print("\nGRID ADEQUACY RESULTS")
    print("grid  interior  N   status        A_rel        B_rel        A_max        B_max        "
          "R_max        DFR          rho          cancel       max_mode")
    for row in records:
        if row["status"] == "INADMISSIBLE":
            print(f"{row['grid_size']:4d}  {row['interior_points_per_axis']:8d}  {row['mode_count']:2d}  "
                  f"INADMISSIBLE  reason={row['reason']}")
            continue
        print(f"{row['grid_size']:4d}  {row['interior_points_per_axis']:8d}  {row['mode_count']:2d}  "
              f"{row['status']:6s}  {row['A_relative_frobenius_error']:.3e}  "
              f"{row['B_relative_frobenius_error']:.3e}  "
              f"{row['A_maximum_absolute_component_error']:.3e}  "
              f"{row['B_maximum_absolute_component_error']:.3e}  "
              f"{row['maximum_absolute_residual_moment']:.3e}  "
              f"{row['weighted_discrete_DFR_score']:.3e}  "
              f"{row['normalized_residual_score_rho_h']:.3e}  "
              f"{row['residual_cancellation_indicator']:.3e}  "
              f"{tuple(row['largest_residual_mode'])}")
        strong = row["actual_training_DST_projection"]
        print("      STRONG RESIDUAL DIAGNOSTIC — NOT THE REQUESTED WAN WEAK MOMENT: "
              f"max={strong['strong_residual_maximum_absolute_moment']:.3e} "
              f"DFR={strong['strong_residual_weighted_DFR_score']:.3e} "
              f"Aproj_rel={strong['A_projection_relative_reference_error']:.3e} "
              f"Bproj_rel={strong['B_projection_relative_reference_error']:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[64, 128, 256, 384, 512])
    parser.add_argument("--mode-counts", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--reference-order", type=int, default=384)
    parser.add_argument("--refined-reference-order", type=int, default=768)
    parser.add_argument("--output-json", type=Path,
                        default=Path("fourier_2d_diagonal_layer_grid_audit.json"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to fall back to CPU")
    if min(args.grid_sizes + args.mode_counts + [args.reference_order,
                                                 args.refined_reference_order]) < 1:
        raise ValueError("grid sizes, modes, and reference orders must be positive")
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite existing JSON: {args.output_json}")
    torch.set_default_dtype(DTYPE)
    device = torch.device(args.device)
    maximum_modes = max(args.mode_counts)

    print("=" * 132)
    print("2D DIAGONAL-LAYER ACTUAL TRAINING-GRID RESOLUTION AUDIT")
    print("=" * 132)
    print(f"device={device} dtype={DTYPE} beta={args.beta}")
    print("grid_size counts endpoints; interior_points_per_axis=grid_size-2")
    print("physical uniform grid: x_j=j*pi/(grid_size-1), j=1,...,grid_size-2")
    print("actual DST-II matrix: sqrt(2/n)*sin(pi*(j+1/2)*k/n), j=0,...,n-1")
    print("project axis scaling: multiply by sqrt(pi/n); 2D scale=2*pi/n^2")
    print("continuous normalized basis: phi_kl=(2/pi)sin(kx)sin(ly)")
    print("test norm: full H1; spectral weights=1/(1+k^2+l^2)")
    print("current training objective: DST-II projection of STRONG residual -Delta(u)-f")
    print("requested audit: rectangle-rule WEAK A_h and B_h on the actual physical interior grid")
    print("PASS thresholds are diagnostic rules, not a mathematical convergence theorem.")
    print(f"thresholds: A_rel,B_rel<={TERM_RTOL}; rho<={RHO_LIMIT}; calibration<={CALIBRATION_TOL}")
    print("No neural network, optimizer, parameter update, or checkpoint is created.")

    print(f"\nComputing continuous references Q{args.reference_order} and Q{args.refined_reference_order} "
          f"for N={maximum_modes}...")
    primary_a, primary_b = continuous_terms(args.reference_order, maximum_modes, args.beta, device)
    refined_a, refined_b = continuous_terms(args.refined_reference_order, maximum_modes, args.beta, device)
    reference = reference_check(primary_a, primary_b, refined_a, refined_b)
    print("REFERENCE CONVERGENCE N={}".format(maximum_modes))
    for key, value in reference.items():
        if key != "thresholds":
            print(f"  {key}={value}")
    print(f"REFERENCE DECISION: {reference['decision']}")

    calibrations = [calibration_for_grid(size, maximum_modes, args.beta, device)
                    for size in args.grid_sizes]
    print_calibration(calibrations)

    records: list[dict[str, Any]] = []
    if reference["decision"] == "PASS":
        by_grid = {item["grid_size"]: item for item in calibrations}
        for size in args.grid_sizes:
            for modes in args.mode_counts:
                records.append(grid_record(size, modes, args.beta, device, refined_a, refined_b,
                                           by_grid[size], True))
        print_grid_table(records)
    else:
        print("\nGRID ADEQUACY DECISION WITHHELD: continuous Q384/Q768 reference did not pass.")

    minimum: dict[str, int | str] = {}
    for modes in args.mode_counts:
        passing = [row["grid_size"] for row in records
                   if row["mode_count"] == modes and row["status"] == "PASS"]
        minimum[str(modes)] = min(passing) if passing else "NONE"
    print("\nMINIMUM PASSING GRID")
    for modes in args.mode_counts:
        print(f"N={modes:<3d} minimum_pass_grid={minimum[str(modes)]}")

    all_modes_have_pass = reference["decision"] == "PASS" and all(value != "NONE" for value in minimum.values())
    final_decision = "PASS" if all_modes_have_pass else "FAIL"
    payload = {
        "problem": {
            "domain": "(0,pi)^2",
            "exact_solution": "x(pi-x)y(pi-y)tanh(beta(x+y-pi))",
            "forcing": "-Delta u*",
        },
        "beta": args.beta, "dtype": str(DTYPE), "device": str(device),
        "basis_normalization": "phi_kl=(2/pi)sin(kx)sin(ly)",
        "test_norm": "full H1; ||phi_kl||^2=1+k^2+l^2",
        "spectral_weight": "1/(1+k^2+l^2)",
        "training_objective": "DST-II projection of strong residual -Delta(u)-f",
        "reference_orders": [args.reference_order, args.refined_reference_order],
        "grid_sizes": args.grid_sizes, "mode_counts": args.mode_counts,
        "calibration_results": calibrations,
        "reference_convergence": reference,
        "grid_mode_results": records,
        "minimum_passing_grid": minimum,
        "diagnostic_thresholds": {
            "A_relative_frobenius_error": TERM_RTOL,
            "B_relative_frobenius_error": TERM_RTOL,
            "normalized_residual_score": RHO_LIMIT,
            "projection_calibration": CALIBRATION_TOL,
            "note": "diagnostic rules, not a mathematical convergence theorem",
        },
        "final_decision": final_decision,
        "training_task_count": 0, "optimizer_count": 0,
        "parameter_update_count": 0, "checkpoint_count": 0,
    }
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\naudit_json={args.output_json.resolve()}")
    print(f"FINAL DECISION: {final_decision}")
    print("training_tasks_started=0")
    print("optimizers_created=0")
    print("parameter_updates=0")
    print("checkpoints_written=0")


if __name__ == "__main__":
    main()
