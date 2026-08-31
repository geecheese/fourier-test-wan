#!/usr/bin/env python3
"""Strictly paired, three-seed DFR-in-WAN versus learned-adversary WAN pilot.

The driver creates one trial initialization per (problem, seed), loads that exact
state into both methods, performs common-grid endpoint/refinement audits, and
retains complete per-run artifacts.  It intentionally has no best-run selection.

Objective classification: Original WAN uses neural weak moments.  Fourier-test
WAN uses explicit weak moments for Dirac and strong residual DST-II projection
for the smooth non-Dirac problems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import numpy as np

from experiments.dirac1d.train_fourier_dfr_point_source_lbfgs import SOURCE, split_rule
from baselines.train_paired_wan_point_source_lbfgs import weak_quantities as dirac_wan_quantities
from experiments.poisson2d.train_spectral_wan import PI, TrialNet, dirac_weak_loss, exact, grid, laplacian
from src.methods.wan_spectral_loss import SpectralWANLoss


PROBLEMS = ("dirac1d", "large_gradient1d", "poisson2d")
SEEDS = (42, 2026, 3407)
WIDTH, DEPTH = 48, 3
DTYPE = torch.float64
BOUNDARY_REPRESENTATION = "product_i x_i*(pi-x_i)"
ACTIVATION = "Tanh"
EPS = 1.0e-30


@dataclass(frozen=True)
class Settings:
    device: str
    modes: int = 32
    grid_size: int = 64
    audit_size: int = 257
    refined_audit_size: int = 513
    dfr_steps: int = 1
    dfr_inner: int = 100
    wan_rounds: int = 100
    wan_inner_u: int = 5
    wan_inner_v: int = 5
    dirac_quadrature_per_half: int = 1024
    refinement_rtol: float = 5.0e-3
    refinement_atol: float = 1.0e-10


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def dimensions(problem: str) -> int:
    return 2 if problem == "poisson2d" else 1


def model_spec(problem: str) -> dict[str, Any]:
    dim = dimensions(problem)
    return {
        "class": "train_spectral_wan.TrialNet",
        "input_dimension": dim,
        "width": WIDTH,
        "depth": DEPTH,
        "activation": ACTIVATION,
        "boundary_representation": BOUNDARY_REPRESENTATION,
        "dtype": str(DTYPE),
    }


def new_trial(problem: str, device: str) -> TrialNet:
    return TrialNet(dimensions(problem), WIDTH, DEPTH, boundary_distance=True).to(
        device=device, dtype=DTYPE
    )


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def states_identical(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[k].cpu(), right[k].cpu()) for k in left)


def points_digest(points: torch.Tensor) -> str:
    return hashlib.sha256(points.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def common_metrics(model: torch.nn.Module, problem: str, points: torch.Tensor) -> dict[str, float]:
    x = points.detach().clone().requires_grad_(True)
    prediction = model(x)
    target = exact(x, problem)
    prediction_gradient = torch.autograd.grad(prediction.sum(), x)[0]
    target_gradient = torch.autograd.grad(target.sum(), x)[0]
    interior_size = round(points.shape[0] ** (1 / dimensions(problem)))
    spacing = PI / (interior_size + 1)
    weight = spacing ** dimensions(problem)
    l2_sq = weight * (prediction - target).square().sum()
    ref_sq = weight * target.square().sum()
    seminorm_sq = weight * (prediction_gradient - target_gradient).square().sum()
    values = {
        "relative_l2": float(torch.sqrt(l2_sq / ref_sq).detach()),
        "absolute_l2": float(torch.sqrt(l2_sq).detach()),
        "h1_seminorm": float(torch.sqrt(seminorm_sq).detach()),
        "h1": float(torch.sqrt(l2_sq + seminorm_sq).detach()),
    }
    return values


def finite_dict(values: dict[str, float]) -> bool:
    return all(math.isfinite(value) for value in values.values())


def refinement_audit(model: torch.nn.Module, problem: str, settings: Settings) -> dict[str, Any]:
    primary_points = grid(settings.audit_size, dimensions(problem), settings.device)
    refined_points = grid(settings.refined_audit_size, dimensions(problem), settings.device)
    primary = common_metrics(model, problem, primary_points)
    refined = common_metrics(model, problem, refined_points)
    differences = {key: abs(primary[key] - refined[key]) for key in primary}
    tolerances = {
        key: settings.refinement_atol + settings.refinement_rtol * max(abs(primary[key]), abs(refined[key]))
        for key in primary
    }


def gauss_rule(problem: str, order: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent Gauss--Legendre audit rule; Dirac is split at pi/2."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    if problem == "dirac1d":
        left_x = 0.25 * PI * (nodes + 1.0)
        right_x = 0.5 * PI + 0.25 * PI * (nodes + 1.0)
        x = np.concatenate((left_x, right_x))
        w = np.concatenate((0.25 * PI * weights, 0.25 * PI * weights))
        return (torch.as_tensor(x[:, None], dtype=DTYPE, device=device),
                torch.as_tensor(w[:, None], dtype=DTYPE, device=device))
    x = 0.5 * PI * (nodes + 1.0)
    w = 0.5 * PI * weights
    if problem == "large_gradient1d":
        return (torch.as_tensor(x[:, None], dtype=DTYPE, device=device),
                torch.as_tensor(w[:, None], dtype=DTYPE, device=device))
    xx, yy = np.meshgrid(x, x, indexing="ij")
    ww = np.multiply.outer(w, w)
    points = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)
    return (torch.as_tensor(points, dtype=DTYPE, device=device),
            torch.as_tensor(ww.reshape(-1, 1), dtype=DTYPE, device=device))


def gauss_metrics(model: torch.nn.Module, problem: str, order: int, device: str) -> dict[str, Any]:
    points, weights = gauss_rule(problem, order, device)
    x = points.detach().clone().requires_grad_(True)
    prediction = model(x)
    target = exact(x, problem)
    prediction_gradient = torch.autograd.grad(prediction.sum(), x)[0]
    target_gradient = torch.autograd.grad(target.sum(), x)[0]
    error = prediction - target
    gradient_error = prediction_gradient - target_gradient
    l2_sq = (weights * error.square()).sum()
    reference_sq = (weights * target.square()).sum()
    seminorm_sq = (weights * gradient_error.square().sum(dim=1, keepdim=True)).sum()
    metrics = {
        "absolute_l2": float(torch.sqrt(l2_sq).detach()),
        "relative_l2": float(torch.sqrt(l2_sq / reference_sq).detach()),
        "h1_seminorm": float(torch.sqrt(seminorm_sq).detach()),
        "h1": float(torch.sqrt(l2_sq + seminorm_sq).detach()),
    }
    return {"order": order, "points": int(points.shape[0]), "points_sha256": points_digest(points),
            "weights_sha256": points_digest(weights), "metrics": metrics}


def exact_gauss_selfcheck(problem: str, order: int, device: str) -> dict[str, float]:
    points, weights = gauss_rule(problem, order, device)
    x = points.detach().clone().requires_grad_(True)
    left = exact(x, problem)
    right = exact(x, problem)
    left_gradient = torch.autograd.grad(left.sum(), x, create_graph=True)[0]
    right_gradient = torch.autograd.grad(right.sum(), x)[0]
    l2_sq = (weights * (left - right).square()).sum()
    seminorm_sq = (weights * (left_gradient - right_gradient).square().sum(dim=1, keepdim=True)).sum()
    return {"absolute_l2": float(torch.sqrt(l2_sq)),
            "h1_seminorm": float(torch.sqrt(seminorm_sq)),
            "h1": float(torch.sqrt(l2_sq + seminorm_sq))}


def endpoint_gauss_audit(model: torch.nn.Module, problem: str, device: str,
                         rtol: float = 5.0e-3, atol: float = 1.0e-10) -> dict[str, Any]:
    coarse = gauss_metrics(model, problem, 96, device)
    refined = gauss_metrics(model, problem, 192, device)
    differences = {key: abs(coarse["metrics"][key] - refined["metrics"][key])
                   for key in coarse["metrics"]}
    relative = {key: differences[key] / max(abs(refined["metrics"][key]), EPS)
                for key in differences}
    tolerances = {key: atol + rtol * max(abs(coarse["metrics"][key]),
                                         abs(refined["metrics"][key]))
                  for key in differences}
    finite = finite_dict(coarse["metrics"]) and finite_dict(refined["metrics"])
    passed = finite and all(differences[key] <= tolerances[key] for key in differences)
    return {"audit_type": "independent_gauss_legendre", "problem": problem,
            "passed": passed, "status": "PASS" if passed else "AUDIT_FAIL",
            "q96": coarse, "q192": refined, "absolute_difference": differences,
            "relative_discrepancy": relative, "tolerances": tolerances,
            "exact_selfcheck_q96": exact_gauss_selfcheck(problem, 96, device),
            "exact_selfcheck_q192": exact_gauss_selfcheck(problem, 192, device)}


def composite_gauss_rule(subintervals: int, order_per_subinterval: int,
                         device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed composite Gauss--Legendre rule on (0, pi)."""
    nodes, weights = np.polynomial.legendre.leggauss(order_per_subinterval)
    edges = np.linspace(0.0, PI, subintervals + 1)
    all_points, all_weights = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        all_points.append(midpoint + half_width * nodes)
        all_weights.append(half_width * weights)
    points = np.concatenate(all_points)[:, None]
    quadrature_weights = np.concatenate(all_weights)[:, None]
    return (torch.as_tensor(points, dtype=DTYPE, device=device),
            torch.as_tensor(quadrature_weights, dtype=DTYPE, device=device))


def weighted_error_metrics(model: torch.nn.Module, problem: str, points: torch.Tensor,
                           weights: torch.Tensor) -> dict[str, float]:
    x = points.detach().clone().requires_grad_(True)
    prediction = model(x)
    target = exact(x, problem)
    prediction_gradient = torch.autograd.grad(prediction.sum(), x)[0]
    target_gradient = torch.autograd.grad(target.sum(), x)[0]
    l2_sq = (weights * (prediction - target).square()).sum()
    exact_l2_sq = (weights * target.square()).sum()
    seminorm_sq = (weights * (prediction_gradient - target_gradient).square()).sum()
    return {"absolute_l2": float(torch.sqrt(l2_sq)),
            "relative_l2": float(torch.sqrt(l2_sq / exact_l2_sq)),
            "h1_seminorm": float(torch.sqrt(seminorm_sq)),
            "h1": float(torch.sqrt(l2_sq + seminorm_sq))}


def exact_norms_on_rule(points: torch.Tensor, weights: torch.Tensor) -> dict[str, float]:
    x = points.detach().clone().requires_grad_(True)
    value = exact(x, "large_gradient1d")
    gradient = torch.autograd.grad(value.sum(), x)[0]
    l2 = torch.sqrt((weights * value.square()).sum())
    seminorm = torch.sqrt((weights * gradient.square()).sum())
    return {"l2_norm": float(l2), "h1_seminorm": float(seminorm)}


def high_order_large_gradient_audit(model: torch.nn.Module, device: str,
                                    rtol: float = 5.0e-3,
                                    atol: float = 1.0e-10) -> dict[str, Any]:
    primary_points, primary_weights = composite_gauss_rule(64, 16, device)
    refined_points, refined_weights = composite_gauss_rule(64, 32, device)
    primary_metrics = weighted_error_metrics(model, "large_gradient1d",
                                             primary_points, primary_weights)
    refined_metrics = weighted_error_metrics(model, "large_gradient1d",
                                             refined_points, refined_weights)
    differences = {key: abs(primary_metrics[key] - refined_metrics[key])
                   for key in primary_metrics}
    relative = {key: differences[key] / max(abs(refined_metrics[key]), EPS)
                for key in differences}
    tolerances = {key: atol + rtol * max(abs(primary_metrics[key]),
                                         abs(refined_metrics[key]))
                  for key in differences}
    primary_exact = exact_norms_on_rule(primary_points, primary_weights)
    refined_exact = exact_norms_on_rule(refined_points, refined_weights)
    exact_difference = {key: abs(primary_exact[key] - refined_exact[key])
                        for key in primary_exact}
    exact_relative = {key: exact_difference[key] / max(abs(refined_exact[key]), EPS)
                      for key in exact_difference}
    passed = (finite_dict(primary_metrics) and finite_dict(refined_metrics)
              and all(differences[key] <= tolerances[key] for key in differences))
    return {
        "audit_type": "composite_gauss_legendre_large_gradient_v2",
        "subintervals": 64, "rtol": rtol, "atol": atol, "passed": passed,
        "status": "PASS" if passed else "AUDIT_FAIL",
        "primary": {"order_per_subinterval": 16, "points": 1024,
                    "points_sha256": points_digest(primary_points),
                    "weights_sha256": points_digest(primary_weights),
                    "metrics": primary_metrics},
        "refined": {"order_per_subinterval": 32, "points": 2048,
                    "points_sha256": points_digest(refined_points),
                    "weights_sha256": points_digest(refined_weights),
                    "metrics": refined_metrics},
        "absolute_difference": differences, "relative_discrepancy": relative,
        "tolerances": tolerances,
        "quadrature_selfcheck": {
            "quantity": "nontrivial exact-solution norms",
            "primary": primary_exact, "refined": refined_exact,
            "absolute_difference": exact_difference,
            "relative_discrepancy": exact_relative,
            "analytic_values": None,
            "analytic_note": "No closed-form norm is implemented for x(pi-x)tanh(20(x-pi/2)).",
        },
    }
    passed = finite_dict(primary) and finite_dict(refined) and all(
        differences[key] <= tolerances[key] for key in primary
    )
    return {
        "passed": passed,
        "primary": primary,
        "refined": refined,
        "absolute_differences": differences,
        "tolerances": tolerances,
        "primary_points_sha256": points_digest(primary_points),
        "refined_points_sha256": points_digest(refined_points),
    }


def lbfgs(parameters: Any, inner: int) -> torch.optim.LBFGS:
    return torch.optim.LBFGS(
        parameters, lr=1.0, max_iter=inner, history_size=50,
        line_search_fn="strong_wolfe", tolerance_grad=1.0e-10,
        tolerance_change=1.0e-14,
    )


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def prepare_run(
    pair_dir: Path, method: str, config: dict[str, Any], initial_state: dict[str, torch.Tensor]
) -> Path:
    output = pair_dir / method
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {output}")
    save_json(output / "config.json", config)
    torch.save(initial_state, output / "initial_model.pt")
    reloaded = torch.load(output / "initial_model.pt", map_location="cpu", weights_only=True)
    if not states_identical(initial_state, reloaded):
        raise RuntimeError(f"initial_model.pt round-trip mismatch: {output}")
    return output


def dfr_train(
    problem: str, seed: int, initial_state: dict[str, torch.Tensor], pair_dir: Path, settings: Settings
) -> dict[str, Any]:
    config = {
        "problem": problem, "method": "dfr_in_wan", "trial_seed": seed,
        "test_seed": None, "trial_architecture": model_spec(problem),
        "test_architecture": None, "initial_state_sha256": state_digest(initial_state),
        "modes": settings.modes if dimensions(problem) == 1 else [settings.modes] * 2,
        "training_grid_size": settings.grid_size, "audit_grid_size": settings.audit_size,
        "refined_audit_grid_size": settings.refined_audit_size,
        "optimizer": {"name": "LBFGS", "steps": settings.dfr_steps,
                      "max_iter": settings.dfr_inner, "lr": 1.0,
                      "history_size": 50, "line_search": "strong_wolfe"},
        "settings": asdict(settings),
    }
    output = prepare_run(pair_dir, "dfr_in_wan", config, initial_state)
    model = new_trial(problem, settings.device)
    model.load_state_dict(initial_state)
    if not states_identical(initial_state, cpu_state(model)):
        raise RuntimeError("DFR trial did not load the paired initial state exactly")
    dim = dimensions(problem)
    coordinates = grid(settings.grid_size, dim, settings.device)
    source = coordinates.detach().clone().requires_grad_(True)
    forcing = None if problem == "dirac1d" else -laplacian(exact(source, problem), source).detach()
    loss_fn = SpectralWANLoss(
        settings.modes if dim == 1 else (settings.modes, settings.modes), dim,
        boundary="dirichlet", domain=PI,
    )
    optimizer = lbfgs(model.parameters(), settings.dfr_inner)
    history: list[dict[str, Any]] = [{"stage": "initial", "trial_closures": 0,
                                      "metrics": common_metrics(model, problem, grid(settings.audit_size, dim, settings.device))}]
    trial_calls = 0

    def closure() -> torch.Tensor:
        nonlocal trial_calls
        trial_calls += 1
        optimizer.zero_grad(set_to_none=True)
        points = coordinates.detach().clone().requires_grad_(True)
        if problem == "dirac1d":
            loss = loss_fn(dirac_weak_loss(model, points, settings.modes))
        else:
            residual = -laplacian(model(points), points) - forcing
            loss = loss_fn.from_samples(residual.reshape((settings.grid_size - 2,) * dim))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite DFR objective at closure {trial_calls}")
        loss.backward()
        history.append({"stage": "closure", "trial_closures": trial_calls,
                        "objective": float(loss.detach())})
        return loss

    sync(settings.device)
    started = time.perf_counter()
    for _ in range(settings.dfr_steps):
        optimizer.step(closure)
    sync(settings.device)
    elapsed = time.perf_counter() - started
    audit = endpoint_gauss_audit(model, problem, settings.device,
                                 settings.refinement_rtol, settings.refinement_atol)
    history.append({"stage": "final", "trial_closures": trial_calls, "seconds": elapsed,
                    "metrics": audit["q192"]["metrics"], "gauss_audit_file": "gauss_audit.json"})
    result = {"status": "PASS" if audit["passed"] else "AUDIT_FAIL", "problem": problem,
              "seed": seed, "method": "dfr_in_wan", "metrics": audit["q192"]["metrics"],
              "gauss_audit": audit,
              "wall_clock_seconds": elapsed, "trial_closures": trial_calls,
              "test_closures": 0, "total_closures": trial_calls,
              "initial_state_sha256": state_digest(initial_state)}
    save_json(output / "history.json", history)
    save_json(output / "gauss_audit.json", audit)
    torch.save({"trial": cpu_state(model), "config": config, "result": result}, output / "final_checkpoint.pt")
    return result


def wan_score(
    problem: str, trial: torch.nn.Module, test: torch.nn.Module,
    training_data: Any, settings: Settings,
) -> torch.Tensor:
    return wan_components(problem, trial, test, training_data, settings)["score"]


def wan_components(
    problem: str, trial: torch.nn.Module, test: torch.nn.Module,
    training_data: Any, settings: Settings,
) -> dict[str, torch.Tensor]:
    if problem == "dirac1d":
        quantities = dirac_wan_quantities(trial, test, training_data)
        return {"score": quantities["score"]}
    x, forcing = training_data
    points = x.detach().clone().requires_grad_(True)
    u, v = trial(points), test(points)
    du = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
    dv = torch.autograd.grad(v.sum(), points, create_graph=True)[0]
    weight = (PI / (settings.grid_size - 1)) ** dimensions(problem)
    moment = weight * (du * dv - forcing * v).sum()
    norm2 = weight * (v.square() + dv.square().sum(dim=1, keepdim=True)).sum()
    return {"score": moment.square() / (norm2 + EPS), "numerator": moment,
            "denominator": norm2}


def wan_train(
    problem: str, seed: int, initial_state: dict[str, torch.Tensor], pair_dir: Path, settings: Settings
) -> dict[str, Any]:
    test_seed = seed + 100000
    config = {
        "problem": problem, "method": "original_wan", "trial_seed": seed,
        "test_seed": test_seed, "test_seed_mapping": "trial_seed + 100000",
        "trial_architecture": model_spec(problem), "test_architecture": model_spec(problem),
        "initial_state_sha256": state_digest(initial_state),
        "training_grid_size": settings.grid_size,
        "dirac_quadrature_order_per_half": settings.dirac_quadrature_per_half,
        "audit_grid_size": settings.audit_size, "refined_audit_grid_size": settings.refined_audit_size,
        "optimizer": {"name": "alternating_LBFGS", "rounds": settings.wan_rounds,
                      "inner_u": settings.wan_inner_u, "inner_v": settings.wan_inner_v,
                      "lr": 1.0, "history_size": 50, "line_search": "strong_wolfe"},
        "objectives": {"test": "-log(normalized_weak_score + 1e-30)",
                       "trial": "normalized_weak_score"},
        "settings": asdict(settings),
    }
    output = prepare_run(pair_dir, "original_wan", config, initial_state)
    trial = new_trial(problem, settings.device)
    trial.load_state_dict(initial_state)
    if not states_identical(initial_state, cpu_state(trial)):
        raise RuntimeError("WAN trial did not load the paired initial state exactly")
    torch.manual_seed(test_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(test_seed)
    test = new_trial(problem, settings.device)
    torch.save(cpu_state(test), output / "initial_test.pt")
    if problem == "dirac1d":
        training_data = split_rule(settings.dirac_quadrature_per_half, settings.device)
    else:
        x = grid(settings.grid_size, dimensions(problem), settings.device)
        source = x.detach().clone().requires_grad_(True)
        forcing = -laplacian(exact(source, problem), source).detach()
        training_data = (x, forcing)
    history: list[dict[str, Any]] = [{"round": 0, "stage": "initial", "trial_closures": 0,
                                      "test_closures": 0,
                                      "metrics": common_metrics(trial, problem, grid(settings.audit_size, dimensions(problem), settings.device))}]
    trial_calls = test_calls = 0
    sync(settings.device)
    started = time.perf_counter()
    for round_number in range(1, settings.wan_rounds + 1):
        for parameter in trial.parameters(): parameter.requires_grad_(False)
        for parameter in test.parameters(): parameter.requires_grad_(True)
        optimizer_v = lbfgs(test.parameters(), settings.wan_inner_v)

        def closure_v() -> torch.Tensor:
            nonlocal test_calls
            test_calls += 1
            optimizer_v.zero_grad(set_to_none=True)
            score = wan_score(problem, trial, test, training_data, settings)
            loss = -torch.log(score + EPS)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite WAN test objective at round {round_number}")
            loss.backward()
            return loss

        optimizer_v.step(closure_v)
        for parameter in test.parameters(): parameter.requires_grad_(False)
        for parameter in trial.parameters(): parameter.requires_grad_(True)
        optimizer_u = lbfgs(trial.parameters(), settings.wan_inner_u)

        def closure_u() -> torch.Tensor:
            nonlocal trial_calls
            trial_calls += 1
            optimizer_u.zero_grad(set_to_none=True)
            loss = wan_score(problem, trial, test, training_data, settings)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite WAN trial objective at round {round_number}")
            loss.backward()
            return loss

        optimizer_u.step(closure_u)
        if round_number == 1 or round_number % 10 == 0 or round_number == settings.wan_rounds:
            metrics = common_metrics(trial, problem, grid(settings.audit_size, dimensions(problem), settings.device))
            if not finite_dict(metrics):
                raise FloatingPointError(f"non-finite WAN metrics at round {round_number}")
            history.append({"round": round_number, "stage": "accepted",
                            "trial_closures": trial_calls, "test_closures": test_calls,
                            "total_closures": trial_calls + test_calls, "metrics": metrics})
    sync(settings.device)
    elapsed = time.perf_counter() - started
    for parameter in test.parameters(): parameter.requires_grad_(True)
    audit = endpoint_gauss_audit(trial, problem, settings.device,
                                 settings.refinement_rtol, settings.refinement_atol)
    history.append({"round": settings.wan_rounds, "stage": "final", "seconds": elapsed,
                    "trial_closures": trial_calls, "test_closures": test_calls,
                    "total_closures": trial_calls + test_calls,
                    "metrics": audit["q192"]["metrics"], "gauss_audit_file": "gauss_audit.json"})
    result = {"status": "PASS" if audit["passed"] else "AUDIT_FAIL", "problem": problem,
              "seed": seed, "method": "original_wan", "metrics": audit["q192"]["metrics"],
              "gauss_audit": audit,
              "wall_clock_seconds": elapsed, "trial_closures": trial_calls,
              "test_closures": test_calls, "total_closures": trial_calls + test_calls,
              "initial_state_sha256": state_digest(initial_state), "test_seed": test_seed}
    save_json(output / "history.json", history)
    save_json(output / "gauss_audit.json", audit)
    torch.save({"trial": cpu_state(trial), "test": cpu_state(test), "config": config,
                "result": result}, output / "final_checkpoint.pt")
    return result


def run_logged(function: Callable[[], dict[str, Any]], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "stdout.log"
    try:
        with log_path.open("w", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            result = function()
            print(json.dumps(result, indent=2))
            return result
    except Exception as error:
        failure = {"status": "FAIL", "error": repr(error), "traceback": traceback.format_exc()}
        save_json(output / "failure.json", failure)
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n" + failure["traceback"])
        raise


def assert_pair_integrity(pair_dir: Path, left: dict[str, Any], right: dict[str, Any]) -> None:
    left_state = torch.load(pair_dir / "dfr_in_wan" / "initial_model.pt", map_location="cpu", weights_only=True)
    right_state = torch.load(pair_dir / "original_wan" / "initial_model.pt", map_location="cpu", weights_only=True)
    if not states_identical(left_state, right_state):
        raise RuntimeError("paired initial_model.pt files differ")
    if left["initial_state_sha256"] != right["initial_state_sha256"]:
        raise RuntimeError("paired initial-state digests differ")
    for order in ("q96", "q192"):
        for key in ("points_sha256", "weights_sha256"):
            if left["gauss_audit"][order][key] != right["gauss_audit"][order][key]:
                raise RuntimeError(f"paired Gauss audit rule differs: {order}.{key}")


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sample_std_ddof_1": statistics.stdev(values)}


def summarize(rows: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    for problem in PROBLEMS:
        for seed in SEEDS:
            for method in ("dfr_in_wan", "original_wan"):
                method_dir = root / problem / f"seed_{seed}" / method
                if not endpoint_is_complete(problem, method_dir):
                    raise RuntimeError(f"final summary refuses incomplete endpoint: {method_dir}")
    result: dict[str, Any] = {"seeds": list(SEEDS), "problems": {}}
    for problem in PROBLEMS:
        problem_rows = [row for row in rows if row["problem"] == problem]
        methods: dict[str, Any] = {}
        for method in ("dfr_in_wan", "original_wan"):
            selected = [row for row in problem_rows if row["method"] == method]
            methods[method] = {
                metric: mean_std([row["metrics"][metric] for row in selected])
                for metric in ("relative_l2", "absolute_l2", "h1_seminorm", "h1")
            }
            methods[method]["wall_clock_seconds"] = mean_std([row["wall_clock_seconds"] for row in selected])
        paired = []
        for seed in SEEDS:
            dfr = next(row for row in problem_rows if row["seed"] == seed and row["method"] == "dfr_in_wan")
            wan = next(row for row in problem_rows if row["seed"] == seed and row["method"] == "original_wan")
            paired.append({"seed": seed,
                           "wan_over_dfr_relative_l2": wan["metrics"]["relative_l2"] / dfr["metrics"]["relative_l2"],
                           "wan_over_dfr_time": wan["wall_clock_seconds"] / dfr["wall_clock_seconds"]})
        result["problems"][problem] = {"methods": methods, "paired_ratios": paired,
                                        "paired_ratio_statistics": {
                                            "wan_over_dfr_relative_l2": mean_std([x["wan_over_dfr_relative_l2"] for x in paired]),
                                            "wan_over_dfr_time": mean_std([x["wan_over_dfr_time"] for x in paired])}}
    return result


def markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = ["# Strictly paired three-seed pilot", "", "No best-seed or best-checkpoint selection was performed.", "",
             "## Raw endpoints", "", "| Problem | Seed | Method | Rel L2 | Abs L2 | H1 semi | H1 | Seconds | Trial/Test/Total closures | Status |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        m = row["metrics"]
        lines.append(f"| {row['problem']} | {row['seed']} | {row['method']} | {m['relative_l2']:.10e} | {m['absolute_l2']:.10e} | {m['h1_seminorm']:.10e} | {m['h1']:.10e} | {row['wall_clock_seconds']:.6f} | {row['trial_closures']}/{row['test_closures']}/{row['total_closures']} | {row['status']} |")
    lines += ["", "## Mean and sample standard deviation (ddof=1)", ""]
    for problem in PROBLEMS:
        lines += [f"### {problem}", ""]
        for method, stats in summary["problems"][problem]["methods"].items():
            lines.append(f"- {method}: " + "; ".join(f"{key}={value['mean']:.10e} ± {value['sample_std_ddof_1']:.10e}" for key, value in stats.items()))
        ratios = summary["problems"][problem]["paired_ratio_statistics"]
        lines.append(f"- WAN/DFR paired Relative-L2 ratio: {ratios['wan_over_dfr_relative_l2']['mean']:.10e} ± {ratios['wan_over_dfr_relative_l2']['sample_std_ddof_1']:.10e}")
        lines.append(f"- WAN/DFR paired time ratio: {ratios['wan_over_dfr_time']['mean']:.10e} ± {ratios['wan_over_dfr_time']['sample_std_ddof_1']:.10e}")
        lines.append("")
    return "\n".join(lines)


class ExactModule(torch.nn.Module):
    def __init__(self, problem: str) -> None:
        super().__init__()
        self.problem = problem

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return exact(points, self.problem)


def cuda_warmup(device: str) -> None:
    if device != "cuda":
        return
    torch.manual_seed(9173)
    temporary = new_trial("poisson2d", device)
    points = torch.rand(256, 2, device=device, dtype=DTYPE) * PI
    value = temporary(points).square().mean()
    value.backward()
    sync(device)
    del temporary, points, value


def spectral_exact_selfcheck(problem: str, settings: Settings) -> dict[str, Any]:
    dim = dimensions(problem)
    points = grid(settings.grid_size, dim, settings.device)
    loss_fn = SpectralWANLoss(
        settings.modes if dim == 1 else (settings.modes, settings.modes), dim,
        boundary="dirichlet", domain=PI, shift=1.0,
    )
    if problem == "dirac1d":
        moments = dirac_weak_loss(ExactModule(problem), points, settings.modes)
    else:
        source = points.detach().clone().requires_grad_(True)
        forcing = -laplacian(exact(source, problem), source).detach()
        evaluation_points = points.detach().clone().requires_grad_(True)
        residual = -laplacian(exact(evaluation_points, problem), evaluation_points) - forcing
        moments = loss_fn.project(residual.reshape((settings.grid_size - 2,) * dim))
    weights = loss_fn.weights(moments.dtype, moments.device)
    weighted_norm = loss_fn(moments)
    return {
        "max_abs_fourier_moment": float(moments.detach().abs().max()),
        "weighted_squared_dual_norm": float(weighted_norm.detach()),
        "finite": bool(torch.isfinite(moments).all() and torch.isfinite(weighted_norm)),
        "spectral_weight_formula": "1/(1+lambda_k)",
        "spectral_weight_min": float(weights.detach().min()),
        "spectral_weight_max": float(weights.detach().max()),
        "spectral_weights": weights.detach().cpu().tolist(),
    }


def neural_exact_selfcheck(problem: str, seed: int, settings: Settings) -> dict[str, Any]:
    test_seed = seed + 100000
    torch.manual_seed(test_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(test_seed)
    test = new_trial(problem, settings.device)
    if problem == "dirac1d":
        points, weights = split_rule(settings.dirac_quadrature_per_half, settings.device)
        x = points.detach().clone().requires_grad_(True)
        v = test(x)
        dv = torch.autograd.grad(v.sum(), x, create_graph=False)[0]
        exact_gradient = torch.where(x < SOURCE, torch.full_like(x, 0.5), torch.full_like(x, -0.5))
        source_point = torch.full((1, 1), SOURCE, dtype=DTYPE, device=settings.device)
        numerator = (weights * exact_gradient * dv).sum() - test(source_point)[0, 0]
        denominator = (weights * (v.square() + dv.square())).sum()
    else:
        points = grid(settings.grid_size, dimensions(problem), settings.device)
        x = points.detach().clone().requires_grad_(True)
        u = exact(x, problem)
        du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        v = test(x)
        dv = torch.autograd.grad(v.sum(), x, create_graph=False)[0]
        forcing_points = points.detach().clone().requires_grad_(True)
        forcing = -laplacian(exact(forcing_points, problem), forcing_points).detach()
        weight = (PI / (settings.grid_size - 1)) ** dimensions(problem)
        numerator = weight * (du * dv - forcing * v).sum()
        denominator = weight * (v.square() + dv.square().sum(dim=1, keepdim=True)).sum()
    normalized_score = numerator.square() / (denominator + EPS)
    return {
        "test_seed": test_seed,
        "test_initialization_sha256": state_digest(cpu_state(test)),
        "weak_numerator": float(numerator.detach()),
        "h1_denominator": float(denominator.detach()),
        "normalized_weak_score": float(normalized_score.detach()),
        "finite": bool(torch.isfinite(numerator) and torch.isfinite(denominator)
                       and torch.isfinite(normalized_score)),
    }


def preflight(settings: Settings) -> None:
    cuda_warmup(settings.device)
    deterministic = {
        "torch_default_dtype": str(torch.get_default_dtype()),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cuda_warmup_forward_backward": settings.device == "cuda",
        "optimizer_steps": 0,
        "parameter_updates": 0,
    }
    print("=" * 96)
    print("STRICTLY PAIRED MULTISEED BENCHMARK — READ-ONLY PREFLIGHT")
    print("=" * 96)
    print("training_output_directory_created: false")
    print("training_tasks_started: 0")
    print("test_space_norm_match: H1")
    print("DFR spectral weights: 1/(1+lambda_k)")
    print("WAN denominator: integral(v^2 + |grad v|^2)")
    print("norm_compatibility: PASS")
    print(f"deterministic_settings: {json.dumps(deterministic, sort_keys=True)}")
    print(f"planned_seed_order: {list(SEEDS)}")
    print("planned_method_order_by_seed: 42=[DFR,WAN], 2026=[WAN,DFR], 3407=[DFR,WAN]")
    all_passed = True
    for problem in PROBLEMS:
        dim = dimensions(problem)
        primary = grid(settings.audit_size, dim, settings.device)
        refined = grid(settings.refined_audit_size, dim, settings.device)
        spectral_check = spectral_exact_selfcheck(problem, settings)
        print("-" * 96)
        print(f"problem: {problem}")
        print("DFR_numerator: Fourier coefficients of weak/strong residual pairing")
        print("DFR_denominator_induced_by_weights: H1 test norm")
        print("DFR_spectral_weights: 1/(1+lambda_k), lambda_k=k^2 (1D) or kx^2+ky^2 (2D)")
        print("WAN_numerator: integral(grad(u) dot grad(v) - f*v); dirac subtracts v(pi/2)")
        print("WAN_denominator: integral(v^2 + |grad(v)|^2)")
        print(f"architecture: {json.dumps(model_spec(problem), sort_keys=True)}")
        print(f"audit_grid_sha256: {points_digest(primary)}")
        print(f"refined_audit_grid_sha256: {points_digest(refined)}")
        print(f"fourier_exact_selfcheck: {json.dumps(spectral_check, sort_keys=True)}")
        all_passed = all_passed and spectral_check["finite"]
        for seed in SEEDS:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            trial_state = cpu_state(new_trial(problem, "cpu"))
            trial_sha = state_digest(trial_state)
            neural_check = neural_exact_selfcheck(problem, seed, settings)
            check_passed = neural_check["finite"] and neural_check["h1_denominator"] > 0.0
            all_passed = all_passed and check_passed
            print(f"seed={seed} trial_initialization_sha256={trial_sha}")
            print(f"seed={seed} fixed_neural_test_selfcheck={json.dumps(neural_check, sort_keys=True)}")
    print("=" * 96)
    print(f"preflight_status: {'PASS' if all_passed else 'FAIL'}")
    print("training_tasks_started: 0")
    if not all_passed:
        raise RuntimeError("preflight failed")


def load_endpoint_model(checkpoint_path: Path, problem: str, device: str) -> tuple[TrialNet, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "trial" not in checkpoint:
        raise KeyError(f"checkpoint has no trial state: {checkpoint_path}")
    model = new_trial(problem, device)
    model.load_state_dict(checkpoint["trial"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def run_reaudit(root: Path, device: str) -> None:
    if not root.is_dir():
        raise FileNotFoundError(root)
    checkpoints = sorted(root.glob("*/seed_*/*/final_checkpoint.pt"))
    if not checkpoints:
        raise RuntimeError(f"no final checkpoints found below {root}")
    print("=" * 108)
    print("GAUSS--LEGENDRE ENDPOINT RE-AUDIT (NO OPTIMIZER, NO PARAMETER UPDATES)")
    print("=" * 108)
    print(f"root={root}")
    print(f"checkpoint_count={len(checkpoints)}")
    print("rules: dirac1d=split Q96/Q192 per half; large_gradient1d=Q96/Q192; poisson2d=Q96xQ96/Q192xQ192")
    for checkpoint_path in checkpoints:
        method_dir = checkpoint_path.parent
        problem = method_dir.parent.parent.name
        seed = int(method_dir.parent.name.removeprefix("seed_"))
        method = method_dir.name
        audit_path = method_dir / "gauss_audit.json"
        if audit_path.exists():
            print(f"SKIP existing (not overwritten): {audit_path}")
            continue
        model, checkpoint = load_endpoint_model(checkpoint_path, problem, device)
        audit = endpoint_gauss_audit(model, problem, device)
        old_status = checkpoint.get("result", {}).get("status", "UNKNOWN")
        audit.update({"problem": problem, "seed": seed, "method": method,
                      "checkpoint": str(checkpoint_path.resolve()),
                      "original_status_preserved": old_status,
                      "reaudit_status": ("PASS_REAUDITED" if audit["passed"] and old_status != "PASS"
                                         else audit["status"]),
                      "optimizer_loaded": False, "parameter_updates": 0})
        save_json(audit_path, audit)
        print("-" * 108)
        print(f"problem={problem} seed={seed} method={method}")
        print(f"original_status={old_status} reaudit_status={audit['reaudit_status']} passed={audit['passed']}")
        print(f"q96_points={audit['q96']['points']} q192_points={audit['q192']['points']}")
        print(f"q96_points_sha256={audit['q96']['points_sha256']}")
        print(f"q192_points_sha256={audit['q192']['points_sha256']}")
        for key in ("absolute_l2", "relative_l2", "h1_seminorm", "h1"):
            print(f"{key}: Q96={audit['q96']['metrics'][key]:.17e} "
                  f"Q192={audit['q192']['metrics'][key]:.17e} "
                  f"absolute_difference={audit['absolute_difference'][key]:.17e} "
                  f"relative_discrepancy={audit['relative_discrepancy'][key]:.17e}")
        print(f"exact_selfcheck_Q96={json.dumps(audit['exact_selfcheck_q96'], sort_keys=True)}")
        print(f"exact_selfcheck_Q192={json.dumps(audit['exact_selfcheck_q192'], sort_keys=True)}")
        print(f"saved={audit_path.resolve()}")
    print("=" * 108)
    print("re-audit complete; training_tasks_started=0 optimizer_loaded=false parameter_updates=0")


def run_large_gradient_high_order_reaudit(root: Path, device: str) -> None:
    problem_root = root / "large_gradient1d"
    checkpoints = sorted(problem_root.glob("seed_*/*/final_checkpoint.pt"))
    if len(checkpoints) != 6:
        raise RuntimeError(f"expected exactly 6 large_gradient1d endpoints, found {len(checkpoints)}")
    print("=" * 112)
    print("LARGE-GRADIENT HIGH-ORDER COMPOSITE GAUSS--LEGENDRE RE-AUDIT V2")
    print("=" * 112)
    print(f"root={root}")
    print("rule_primary=64_subintervals_x_16_points=1024")
    print("rule_refined=64_subintervals_x_32_points=2048")
    print("rtol=0.005 atol=1e-10")
    shared_hashes: dict[str, str] | None = None
    for checkpoint_path in checkpoints:
        method_dir = checkpoint_path.parent
        audit_path = method_dir / "gauss_audit_v2.json"
        if audit_path.exists():
            raise FileExistsError(f"refusing to overwrite existing audit: {audit_path}")
        seed = int(method_dir.parent.name.removeprefix("seed_"))
        method = method_dir.name
        model, checkpoint = load_endpoint_model(checkpoint_path, "large_gradient1d", device)
        audit = high_order_large_gradient_audit(model, device)
        hashes = {"primary_points": audit["primary"]["points_sha256"],
                  "primary_weights": audit["primary"]["weights_sha256"],
                  "refined_points": audit["refined"]["points_sha256"],
                  "refined_weights": audit["refined"]["weights_sha256"]}
        if shared_hashes is None:
            shared_hashes = hashes
        elif hashes != shared_hashes:
            raise RuntimeError("composite quadrature points/weights differ between endpoints")
        audit.update({"problem": "large_gradient1d", "seed": seed, "method": method,
                      "checkpoint": str(checkpoint_path.resolve()),
                      "original_status_preserved": checkpoint.get("result", {}).get("status", "UNKNOWN"),
                      "optimizer_loaded": False, "parameter_updates": 0})
        save_json(audit_path, audit)
        print("-" * 112)
        print(f"seed={seed} method={method} status={audit['status']}")
        print(f"primary_points_sha256={hashes['primary_points']}")
        print(f"primary_weights_sha256={hashes['primary_weights']}")
        print(f"refined_points_sha256={hashes['refined_points']}")
        print(f"refined_weights_sha256={hashes['refined_weights']}")
        for key in ("absolute_l2", "relative_l2", "h1_seminorm", "h1"):
            print(f"{key}: primary={audit['primary']['metrics'][key]:.17e} "
                  f"refined={audit['refined']['metrics'][key]:.17e} "
                  f"absolute_difference={audit['absolute_difference'][key]:.17e} "
                  f"relative_discrepancy={audit['relative_discrepancy'][key]:.17e}")
        check = audit["quadrature_selfcheck"]
        for key in ("l2_norm", "h1_seminorm"):
            print(f"exact_{key}: primary={check['primary'][key]:.17e} "
                  f"refined={check['refined'][key]:.17e} "
                  f"absolute_difference={check['absolute_difference'][key]:.17e} "
                  f"relative_discrepancy={check['relative_discrepancy'][key]:.17e}")
        print(f"analytic_norms={check['analytic_values']} note={check['analytic_note']}")
        print(f"saved={audit_path.resolve()}")
    print("=" * 112)
    print("checkpoint_count=6")
    print("training_tasks_started=0")
    print("optimizer_loaded=false")
    print("parameter_updates=0")


def authoritative_audit_path(problem: str, method_dir: Path) -> Path:
    """Return the sole authoritative endpoint-audit path for a problem."""
    return method_dir / ("gauss_audit_v2.json" if problem == "large_gradient1d"
                         else "gauss_audit.json")


def endpoint_is_complete(problem: str, method_dir: Path) -> bool:
    """Completion is defined only by checkpoint existence and authoritative Gauss pass."""
    if not (method_dir / "final_checkpoint.pt").is_file():
        return False
    audit_path = authoritative_audit_path(problem, method_dir)
    if not audit_path.is_file():
        return False
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    return audit.get("passed") is True


def resume_partition(root: Path) -> tuple[set[tuple[str, int, str]], set[tuple[str, int, str]]]:
    skipped: set[tuple[str, int, str]] = set()
    planned: set[tuple[str, int, str]] = set()
    for problem in PROBLEMS:
        for seed in SEEDS:
            for method in ("dfr_in_wan", "original_wan"):
                key = (problem, seed, method)
                method_dir = root / problem / f"seed_{seed}" / method
                (skipped if endpoint_is_complete(problem, method_dir) else planned).add(key)
    return skipped, planned


def existing_row(problem: str, method_dir: Path) -> dict[str, Any]:
    if not endpoint_is_complete(problem, method_dir):
        raise RuntimeError(f"cannot reconstruct incomplete endpoint: {method_dir}")
    checkpoint = torch.load(method_dir / "final_checkpoint.pt", map_location="cpu", weights_only=True)
    audit = json.loads(authoritative_audit_path(problem, method_dir).read_text(encoding="utf-8"))
    row = dict(checkpoint["result"])
    metrics_container = audit.get("q192", audit.get("refined"))
    row["metrics"] = metrics_container["metrics"]
    row["gauss_audit"] = audit
    row["original_status"] = row.get("status", "UNKNOWN")
    row["status"] = ("PASS_REAUDITED" if audit.get("reaudit_status") == "PASS_REAUDITED"
                     else "PASS")
    row["authoritative_audit_path"] = str(authoritative_audit_path(problem, method_dir).resolve())
    return row


def run_resume_plan_only(root: Path) -> None:
    """Read-only resume planner: inspect paths and JSON, never deserialize checkpoints."""
    if not root.is_dir():
        raise FileNotFoundError(root)
    skipped_keys, planned_keys = resume_partition(root)
    skipped = []
    planned = []
    for problem in PROBLEMS:
        for seed in SEEDS:
            for method in ("dfr_in_wan", "original_wan"):
                key = (problem, seed, method)
                method_dir = root / problem / f"seed_{seed}" / method
                audit_path = authoritative_audit_path(problem, method_dir)
                record = {"problem": problem, "seed": seed, "method": method,
                          "checkpoint_exists": (method_dir / "final_checkpoint.pt").is_file(),
                          "audit_path": str(audit_path.resolve()),
                          "audit_exists": audit_path.is_file(),
                          "audit_passed": endpoint_is_complete(problem, method_dir)}
                (skipped if key in skipped_keys else planned).append(record)
    expected_runs = [
        ("poisson2d", 2026, "dfr_in_wan"),
        ("poisson2d", 2026, "original_wan"),
        ("poisson2d", 3407, "dfr_in_wan"),
        ("poisson2d", 3407, "original_wan"),
    ]
    actual_runs = [(row["problem"], row["seed"], row["method"]) for row in planned]
    print("=" * 96)
    print("PAIRED MULTISEED READ-ONLY RESUME PLAN")
    print("=" * 96)
    for row in skipped:
        print(f"SKIP problem={row['problem']} seed={row['seed']} method={row['method']} "
              f"checkpoint=true audit_passed=true audit={row['audit_path']}")
    for row in planned:
        print(f"RUN  problem={row['problem']} seed={row['seed']} method={row['method']} "
              f"checkpoint={str(row['checkpoint_exists']).lower()} "
              f"audit_exists={str(row['audit_exists']).lower()} "
              f"audit_passed={str(row['audit_passed']).lower()}")
    print("-" * 96)
    print(f"skip_count={len(skipped)}")
    print(f"run_count={len(planned)}")
    print("models_loaded=0")
    print("optimizers_loaded=0")
    print("files_created_or_modified=0")
    if len(skipped) != 14 or len(planned) != 4 or actual_runs != expected_runs:
        raise RuntimeError(f"invalid resume plan: expected 14 skip and exact 4 runs; actual_runs={actual_runs}")
    print("resume_plan_status=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("paired_multiseed_runs"))
    parser.add_argument("--preflight-only", action="store_true",
                        help="run read-only integrity/self-checks without creating run directories")
    parser.add_argument("--reaudit-only", action="store_true",
                        help="audit existing final checkpoints with independent Gauss rules")
    parser.add_argument("--reaudit-large-gradient-high-order", action="store_true",
                        help="read-only composite-Gauss v2 audit of six large-gradient endpoints")
    parser.add_argument("--resume", action="store_true",
                        help="resume after completed Gauss-audited methods without retraining them")
    parser.add_argument("--resume-plan-only", action="store_true",
                        help="read-only inspection of the exact 14-skip/4-run resume plan")
    args = parser.parse_args()
    if args.resume_plan_only:
        run_resume_plan_only(args.output_dir.resolve())
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu")
    torch.set_default_dtype(DTYPE)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    settings = Settings(device=args.device)
    if args.preflight_only:
        preflight(settings)
        return
    root = args.output_dir.resolve()
    if args.reaudit_only:
        run_reaudit(root, args.device)
        return
    if args.reaudit_large_gradient_high_order:
        run_large_gradient_high_order_reaudit(root, args.device)
        return
    if args.resume:
        if not root.is_dir():
            raise FileNotFoundError(root)
        plan_skip_set, plan_run_set = resume_partition(root)
        resume_skip_set, resume_run_set = resume_partition(root)
        if plan_skip_set != resume_skip_set or plan_run_set != resume_run_set:
            raise RuntimeError("resume-plan and formal-resume endpoint sets differ")
        expected_run_set = {("poisson2d", seed, method)
                            for seed in (2026, 3407)
                            for method in ("dfr_in_wan", "original_wan")}
        if len(resume_skip_set) != 14 or resume_run_set != expected_run_set:
            raise RuntimeError(f"formal resume precheck failed before model loading: "
                               f"skip={len(resume_skip_set)} run={sorted(resume_run_set)}")
        print("RESUME CONSISTENCY PASS: plan/formal SKIP sets identical (14 skip, 4 run)")
    else:
        root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        for seed in SEEDS:
            pair_dir = root / problem / f"seed_{seed}"
            if args.resume and pair_dir.is_dir():
                complete_methods = []
                for method in ("dfr_in_wan", "original_wan"):
                    method_dir = pair_dir / method
                    if endpoint_is_complete(problem, method_dir):
                        rows.append(existing_row(problem, method_dir))
                        complete_methods.append(method)
                if len(complete_methods) == 2:
                    print(f"RESUME SKIP {problem} seed={seed}: both methods already Gauss-audited")
                    continue
                if problem != "poisson2d" or seed == 42:
                    raise RuntimeError(f"resume refuses incomplete/failed pre-resume pair: {pair_dir}")
            else:
                if args.resume and (problem != "poisson2d" or seed == 42):
                    raise RuntimeError("resume may create new tasks only from poisson2d seed=2026 onward")
                pair_dir.mkdir(parents=True, exist_ok=False)
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            initial_state = cpu_state(new_trial(problem, "cpu"))
            torch.save(initial_state, pair_dir / "initial_model.pt")
            digest = state_digest(initial_state)
            save_json(pair_dir / "pair_config.json", {"problem": problem, "seed": seed,
                      "test_seed": seed + 100000, "architecture": model_spec(problem),
                      "initial_state_sha256": digest, "settings": asdict(settings)})
            method_order = ("dfr_in_wan", "original_wan") if SEEDS.index(seed) % 2 == 0 else ("original_wan", "dfr_in_wan")
            method_results: dict[str, dict[str, Any]] = {}
            for method in method_order:
                method_dir = pair_dir / method
                if args.resume and endpoint_is_complete(problem, method_dir):
                    method_results[method] = existing_row(problem, method_dir)
                    continue
                function = (lambda: dfr_train(problem, seed, deepcopy(initial_state), pair_dir, settings)) if method == "dfr_in_wan" else (lambda: wan_train(problem, seed, deepcopy(initial_state), pair_dir, settings))
                method_results[method] = run_logged(function, method_dir)
            dfr, wan = method_results["dfr_in_wan"], method_results["original_wan"]
            assert_pair_integrity(pair_dir, dfr, wan)
            save_json(pair_dir / "integrity.json", {"passed": True, "initial_state_sha256": digest,
                      "audit_points_identical": True, "architecture_identical": True,
                      "dtype_identical": True, "boundary_representation_identical": True})
            rows.extend((dfr, wan))
            if dfr["status"] not in {"PASS", "PASS_REAUDITED"} or wan["status"] not in {"PASS", "PASS_REAUDITED"}:
                save_json(root / "partial_results.json", rows)
                raise RuntimeError(f"audit failure for {problem}, seed={seed}; stopping after preservation")
    statistical_summary = summarize(rows, root)
    payload = {"raw_results": rows, "statistics": statistical_summary}
    save_json(root / "paired_multiseed_results.json", payload)
    (root / "paired_multiseed_summary.md").write_text(markdown(rows, statistical_summary), encoding="utf-8")
    print(f"Results: {root / 'paired_multiseed_results.json'}")
    print(f"Summary: {root / 'paired_multiseed_summary.md'}")


if __name__ == "__main__":
    main()
