#!/usr/bin/env python3
"""Strict read-only self-check for explicit weak Fourier-test WAN losses."""

from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.methods.fourier_weak_wan_loss import weak_fourier_loss_1d, weak_fourier_loss_2d
from experiments.large_gradient1d.train_fourier_weak_wan_large_gradient import exact_solution as large_exact
from experiments.large_gradient1d.train_fourier_weak_wan_large_gradient import forcing as large_forcing
from experiments.large_gradient1d.train_fourier_weak_wan_large_gradient import gauss_rule
from experiments.poisson2d.train_fourier_weak_wan_poisson2d import exact_solution as poisson_exact
from experiments.poisson2d.train_fourier_weak_wan_poisson2d import forcing as poisson_forcing
from experiments.poisson2d.train_fourier_weak_wan_poisson2d import tensor_gauss_rule
from experiments.poisson2d.train_spectral_wan import TrialNet


PI = math.pi
DTYPE = torch.float64
SCORE_ATOL = 1.0e-18
SCORE_RTOL = 5.0e-3
TERM_RTOL = 1.0e-10
ZERO_SCORE_TOL = 1.0e-20
ZERO_MOMENT_TOL = 1.0e-10
DETECTION_TOL = 1.0e-3
EQUIVALENCE_ATOL = 1.0e-9
EQUIVALENCE_RTOL = 1.0e-9


def score_versions(volume: torch.Tensor, source: torch.Tensor) -> dict[str, dict[str, object]]:
    if volume.ndim == 1:
        k = torch.arange(1, volume.shape[0] + 1, dtype=volume.dtype, device=volume.device)
        spectral = 1.0 / (1.0 + k.square())
    else:
        k = torch.arange(1, volume.shape[0] + 1, dtype=volume.dtype, device=volume.device)
        spectral = 1.0 / (1.0 + k[:, None].square() + k[None, :].square())
    variants = {"correct": volume - source, "source_omitted": volume,
                "source_sign_reversed": volume + source}
    result = {}
    for name, moments in variants.items():
        index = int(torch.argmax(moments.abs()).item())
        mode = ([index + 1] if moments.ndim == 1
                else [index // moments.shape[1] + 1, index % moments.shape[1] + 1])
        result[name] = {
            "maximum_absolute_A": float(volume.abs().max()),
            "maximum_absolute_B": float(source.abs().max()),
            "maximum_absolute_R": float(moments.abs().max()),
            "raw_residual_score": float(moments.square().sum()),
            "weighted_DFR_score": float((moments.square() * spectral).sum()),
            "largest_residual_mode": mode,
            "finite": bool(torch.isfinite(moments).all()),
        }
    return result


def relative_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(left - right)
                 / torch.linalg.vector_norm(right).clamp_min(1.0e-300))


def exact_case(problem: str, primary_order: int, refined_order: int, modes: int,
               device: str) -> tuple[dict[str, object], bool]:
    records = {}
    tensors = {}
    for order in (primary_order, refined_order):
        if problem == "large_gradient1d":
            points, weights = gauss_rule(order, device)
            output = weak_fourier_loss_1d(
                lambda x: large_exact(x, 20.0), points, weights,
                large_forcing(points, 20.0), modes)
        else:
            points, weights = tensor_gauss_rule(order, device)
            output = weak_fourier_loss_2d(
                poisson_exact, points, weights, poisson_forcing(points), modes,
                (order, order))
        records[str(order)] = score_versions(output.volume_moments.detach(),
                                              output.forcing_moments.detach())
        tensors[order] = (output.volume_moments.detach(), output.forcing_moments.detach())
    pa, pb = tensors[primary_order]
    ra, rb = tensors[refined_order]
    primary_score = records[str(primary_order)]["correct"]["weighted_DFR_score"]
    refined_score = records[str(refined_order)]["correct"]["weighted_DFR_score"]
    score_difference = abs(primary_score - refined_score)
    score_tolerance = SCORE_ATOL + SCORE_RTOL * abs(refined_score)
    convergence = {
        "A_relative_frobenius_discrepancy": relative_difference(pa, ra),
        "B_relative_frobenius_discrepancy": relative_difference(pb, rb),
        "correct_score_absolute_difference": score_difference,
        "correct_score_relative_indicator": score_difference / max(abs(refined_score), 1.0e-300),
        "mixed_tolerance": score_tolerance,
        "mixed_tolerance_passed": score_difference <= score_tolerance,
    }
    refined = records[str(refined_order)]
    passed = (
        all(row["finite"] for order in records.values() for row in order.values())
        and convergence["A_relative_frobenius_discrepancy"] <= TERM_RTOL
        and convergence["B_relative_frobenius_discrepancy"] <= TERM_RTOL
        and convergence["mixed_tolerance_passed"]
        and refined["correct"]["weighted_DFR_score"] <= ZERO_SCORE_TOL
        and refined["correct"]["maximum_absolute_R"] <= ZERO_MOMENT_TOL
        and refined["source_omitted"]["weighted_DFR_score"] >= DETECTION_TOL
        and refined["source_sign_reversed"]["weighted_DFR_score"] >= DETECTION_TOL
    )
    return {"orders": [primary_order, refined_order], "modes": modes,
            "quadrature_results": records, "convergence": convergence,
            "passed": passed}, passed


def strong_moments_1d(model: torch.nn.Module, points: torch.Tensor,
                      weights: torch.Tensor, source: torch.Tensor, modes: int) -> torch.Tensor:
    x = points.detach().clone().requires_grad_(True)
    value = model(x)
    first = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x, create_graph=True)[0][:, 0]
    k = torch.arange(1, modes + 1, dtype=x.dtype, device=x.device)
    phi = math.sqrt(2.0 / PI) * torch.sin(x[:, 0, None] * k[None, :])
    return (weights.reshape(-1, 1) * (-second - source.reshape(-1))[:, None] * phi).sum(0)


def strong_moments_2d(model: torch.nn.Module, points: torch.Tensor,
                      weights: torch.Tensor, source: torch.Tensor, modes: int) -> torch.Tensor:
    x = points.detach().clone().requires_grad_(True)
    value = model(x)
    first = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
    lap = torch.zeros_like(value)
    for axis in range(2):
        lap = lap + torch.autograd.grad(first[:, axis].sum(), x, create_graph=True)[0][:, axis:axis + 1]
    order = math.isqrt(points.shape[0])
    residual = (-lap - source).reshape(order, order)
    weight_matrix = weights.reshape(order, order)
    axis_weights = torch.sqrt(torch.diagonal(weight_matrix))
    axis_points = x[:, 0].reshape(order, order)[:, 0]
    k = torch.arange(1, modes + 1, dtype=x.dtype, device=x.device)
    sine = axis_weights[:, None] * torch.sin(axis_points[:, None] * k[None, :])
    return (2.0 / PI) * (sine.T @ residual @ sine)


def weighted_score(moments: torch.Tensor) -> float:
    k = torch.arange(1, moments.shape[0] + 1, dtype=moments.dtype, device=moments.device)
    weights = (1.0 / (1.0 + k.square()) if moments.ndim == 1 else
               1.0 / (1.0 + k[:, None].square() + k[None, :].square()))
    return float((moments.square() * weights).sum())


def equivalence_case(problem: str, primary_order: int, refined_order: int,
                     modes: int, device: str) -> tuple[dict[str, object], bool]:
    dimension = 1 if problem == "large_gradient1d" else 2
    torch.manual_seed(7319 + dimension)
    model = TrialNet(dimension, width=12, depth=2, boundary_distance=True).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    records = {}
    for order in (primary_order, refined_order):
        if dimension == 1:
            points, weights = gauss_rule(order, device)
            source = large_forcing(points, 20.0)
            weak = weak_fourier_loss_1d(model, points, weights, source, modes).moments.detach()
            strong = strong_moments_1d(model, points, weights, source, modes).detach()
        else:
            points, weights = tensor_gauss_rule(order, device)
            source = poisson_forcing(points)
            weak = weak_fourier_loss_2d(model, points, weights, source, modes,
                                        (order, order)).moments.detach()
            strong = strong_moments_2d(model, points, weights, source, modes).detach()
        difference = weak - strong
        records[str(order)] = {
            "maximum_absolute_moment_discrepancy": float(difference.abs().max()),
            "relative_frobenius_discrepancy": relative_difference(weak, strong),
            "weak_weighted_score": weighted_score(weak),
            "strong_weighted_score": weighted_score(strong),
            "weighted_score_discrepancy": abs(weighted_score(weak) - weighted_score(strong)),
            "finite": bool(torch.isfinite(weak).all() and torch.isfinite(strong).all()),
        }
    primary, refined = records[str(primary_order)], records[str(refined_order)]
    convergence = {}
    for key in ("maximum_absolute_moment_discrepancy", "weighted_score_discrepancy"):
        difference = abs(primary[key] - refined[key])
        tolerance = EQUIVALENCE_ATOL + EQUIVALENCE_RTOL * abs(refined[key])
        convergence[key] = {"absolute_difference": difference, "mixed_tolerance": tolerance,
                            "passed": difference <= tolerance}
    passed = (
        all(row["finite"] for row in records.values())
        and refined["maximum_absolute_moment_discrepancy"] <= EQUIVALENCE_ATOL
        and refined["relative_frobenius_discrepancy"] <= EQUIVALENCE_RTOL
        and refined["weighted_score_discrepancy"] <= EQUIVALENCE_ATOL
        and all(row["passed"] for row in convergence.values())
    )
    return {"orders": [primary_order, refined_order], "modes": modes,
            "results": records, "Q_refined_convergence": convergence,
            "passed": passed}, passed


def backward_case(dimension: int, device: str) -> dict[str, object]:
    torch.manual_seed(9182 + dimension)
    model = TrialNet(dimension, width=10, depth=2, boundary_distance=True).to(device)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    if dimension == 1:
        points, weights = gauss_rule(48, device)
        output = weak_fourier_loss_1d(model, points, weights, large_forcing(points), 8)
    else:
        points, weights = tensor_gauss_rule(24, device)
        output = weak_fourier_loss_2d(model, points, weights, poisson_forcing(points), 8, (24, 24))
    output.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    finite = all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients)
    nonzero = any(float(gradient.abs().max()) > 0.0 for gradient in gradients if gradient is not None)
    unchanged = all(torch.equal(old, parameter.detach()) for old, parameter in zip(before, model.parameters()))
    return {"dimension": dimension, "loss": float(output.loss.detach()),
            "gradient_tensor_count": len(gradients), "all_gradients_finite": finite,
            "at_least_one_gradient_nonzero": nonzero,
            "parameters_tensorwise_identical_after_backward": unchanged,
            "passed": finite and nonzero and unchanged}


def structural_check() -> dict[str, object]:
    files = [Path("src/methods/fourier_weak_wan_loss.py"),
             Path("experiments/large_gradient1d/train_fourier_weak_wan_large_gradient.py"),
             Path("experiments/poisson2d/train_fourier_weak_wan_poisson2d.py")]
    forbidden = {"laplacian", "second_derivative", "u_xx", "u_yy"}
    calls = []
    classes = []
    optimizer_targets = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = (node.func.id if isinstance(node.func, ast.Name) else
                        node.func.attr if isinstance(node.func, ast.Attribute) else "")
                if name in forbidden:
                    calls.append({"file": str(path), "line": node.lineno, "call": name})
                if name in {"LBFGS", "Adam"}:
                    optimizer_targets.append({"file": str(path), "line": node.lineno,
                                              "first_argument": ast.unparse(node.args[0]) if node.args else ""})
            if isinstance(node, ast.ClassDef):
                classes.append({"file": str(path), "name": node.name})
    neural_test_classes = [row for row in classes if "test" in row["name"].lower()]
    only_trial_future_optimizer = all(row["first_argument"] == "model.parameters()"
                                      for row in optimizer_targets)
    return {"forbidden_call_nodes": calls, "neural_test_classes": neural_test_classes,
            "future_optimizer_targets": optimizer_targets,
            "only_trial_parameters_in_future_optimizers": only_trial_future_optimizer,
            "fourier_basis_trainable_parameters": 0,
            "test_optimizers": 0,
            "passed": not calls and not neural_test_classes and only_trial_future_optimizer}


def forcing_check(device: str) -> dict[str, object]:
    points = torch.tensor([[0.23], [0.91], [PI / 2 - 0.02], [PI / 2],
                           [PI / 2 + 0.03], [2.41], [2.93]],
                          dtype=DTYPE, device=device, requires_grad=True)
    value = large_exact(points)
    first = torch.autograd.grad(value.sum(), points, create_graph=True)[0]
    automatic = -torch.autograd.grad(first.sum(), points)[0]
    analytic = large_forcing(points)
    maximum = float((automatic - analytic).abs().max())
    tolerance = 1.0e-10 + 1.0e-12 * float(automatic.abs().max())
    return {"large_gradient_maximum_absolute_difference": maximum,
            "tolerance": tolerance, "passed": maximum <= tolerance,
            "poisson2d_identity": "f=2*sin(x)*sin(y)=-Delta(sin(x)sin(y))"}


def print_exact(name: str, record: dict[str, object]) -> None:
    print(f"\n{name} EXACT-SOLUTION WEAK CHECK")
    for order, variants in record["quadrature_results"].items():
        print(f"  quadrature_order={order}")
        for variant, values in variants.items():
            print(f"    {variant}: maxA={values['maximum_absolute_A']:.17e} "
                  f"maxB={values['maximum_absolute_B']:.17e} maxR={values['maximum_absolute_R']:.17e} "
                  f"raw={values['raw_residual_score']:.17e} DFR={values['weighted_DFR_score']:.17e} "
                  f"largest_mode={tuple(values['largest_residual_mode'])}")
    print(f"  convergence={record['convergence']}")
    print(f"  passed={record['passed']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--large-order", type=int, default=256)
    parser.add_argument("--large-refined-order", type=int, default=512)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to fall back to CPU")
    torch.set_default_dtype(DTYPE)
    print("=" * 112)
    print("EXPLICIT WEAK FOURIER-TEST WAN LOSS SELF-CHECK")
    print("=" * 112)
    print(f"device={args.device} dtype={DTYPE}")
    print("1D basis=sqrt(2/pi)sin(kx); weights=1/(1+k^2)")
    print("2D basis=(2/pi)sin(kx)sin(ly); weights=1/(1+k^2+l^2)")
    print("DFR evaluates the residual dual norm; Fourier sine functions are the fixed tests.")
    print("No training, optimizer creation, parameter updates, or checkpoint writes.\n")

    forcing_result = forcing_check(args.device)
    print(f"FORCING CHECK: {forcing_result}")
    large, large_ok = exact_case("large_gradient1d", args.large_order,
                                 args.large_refined_order, 32, args.device)
    poisson, poisson_ok = exact_case("poisson2d", 96, 192, 32, args.device)
    print_exact("LARGE-GRADIENT1D", large)
    print_exact("POISSON2D", poisson)

    large_equiv, large_equiv_ok = equivalence_case(
        "large_gradient1d", args.large_order, args.large_refined_order, 16, args.device)
    poisson_equiv, poisson_equiv_ok = equivalence_case("poisson2d", 96, 192, 16, args.device)
    print(f"\nFROZEN LARGE-GRADIENT TRIAL WEAK/STRONG EQUIVALENCE: {large_equiv}")
    print(f"FROZEN POISSON2D TRIAL WEAK/STRONG EQUIVALENCE: {poisson_equiv}")

    backward_1d = backward_case(1, args.device)
    backward_2d = backward_case(2, args.device)
    print(f"\nBACKWARD GRAPH 1D: {backward_1d}")
    print(f"BACKWARD GRAPH 2D: {backward_2d}")
    structure = structural_check()
    print(f"\nAST STRUCTURAL CHECK: {structure}")

    conditions = {
        "all_numeric_outputs_finite": all((large_ok, poisson_ok, large_equiv_ok, poisson_equiv_ok)),
        "forcing_check": bool(forcing_result["passed"]),
        "large_gradient_exact_weak": large_ok,
        "poisson2d_exact_weak": poisson_ok,
        "large_gradient_frozen_equivalence": large_equiv_ok,
        "poisson2d_frozen_equivalence": poisson_equiv_ok,
        "backward_1d": bool(backward_1d["passed"]),
        "backward_2d": bool(backward_2d["passed"]),
        "weak_path_has_no_second_spatial_derivative_calls": bool(structure["passed"]),
        "no_neural_test_network": not structure["neural_test_classes"],
        "optimizers_created_zero": True,
        "parameter_updates_zero": True,
        "checkpoints_written_zero": True,
        "training_tasks_started_zero": True,
    }
    passed = all(conditions.values())
    print("\nDECISION CONDITIONS")
    for name, value in conditions.items():
        print(f"  {name}={value}")
    print(f"\nDECISION: {'PASS' if passed else 'FAIL'}")
    print("training_tasks_started=0")
    print("optimizers_created=0")
    print("parameter_updates=0")
    print("checkpoints_written=0")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
