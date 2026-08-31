#!/usr/bin/env python3
"""Paired learned-adversary WAN baseline for the 1D Fourier-DFR pilot.

The trial starts from exactly the same checkpoint as DFR and strong PINN.
The neural test function maximizes the normalized weak residual with a log
objective; the trial minimizes the raw normalized weak score.  Q/Q-refined
audits guard every accepted alternating round.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch

from src.common.train_fourier_dfr_1d_lbfgs import (
    TrialNet,
    derivative,
    diagnostics,
    fourier_data,
    gauss_rule,
)


EPS = 1.0e-30


def set_trainable(model: TrialNet, value: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(value)


def weak_quantities(
    trial: TrialNet,
    test: TrialNet,
    rule: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    coordinates, weights = rule
    x = coordinates.detach().clone().requires_grad_(True)
    u = trial(x)
    v = test(x)
    du = derivative(u, x)
    dv = derivative(v, x)
    forcing = 4.0 * torch.sin(2.0 * x)
    moment = (weights * (du * dv - forcing * v)).sum()
    norm2 = (weights * (v.square() + dv.square())).sum()
    score = moment.square() / (norm2 + EPS)

    exact_derivative = 2.0 * torch.cos(2.0 * x)
    exact_moment = (weights * (exact_derivative * dv - forcing * v)).sum()
    exact_normalized_abs = exact_moment.abs() / torch.sqrt(norm2 + EPS)
    return {
        "score": score,
        "moment": moment,
        "norm2": norm2,
        "exact_normalized_abs": exact_normalized_abs,
    }


def max_parameter(model: TrialNet) -> float:
    return max(float(p.detach().abs().max()) for p in model.parameters())


def audit(
    trial: TrialNet,
    test: TrialNet,
    train_rule: tuple[torch.Tensor, torch.Tensor],
    refined_rule: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float | bool]:
    q = weak_quantities(trial, test, train_rule)
    refined = weak_quantities(trial, test, refined_rule)
    score_q = float(q["score"].detach())
    score_refined = float(refined["score"].detach())
    absolute_gap = abs(score_q - score_refined)
    relative_gap = absolute_gap / max(score_q, score_refined, EPS)
    exact_abs = max(
        float(q["exact_normalized_abs"].detach()),
        float(refined["exact_normalized_abs"].detach()),
    )
    parameter_max = max_parameter(test)
    finite = all(
        math.isfinite(value)
        for value in (score_q, score_refined, relative_gap, exact_abs, parameter_max)
    )
    passed = (
        finite
        and (relative_gap <= 1.0e-6 or absolute_gap <= 1.0e-12)
        and exact_abs <= 1.0e-8
        and parameter_max <= 1.0e4
    )
    return {
        "passed": passed,
        "score_q": score_q,
        "score_refined": score_refined,
        "absolute_gap": absolute_gap,
        "relative_gap": relative_gap,
        "exact_normalized_abs": exact_abs,
        "max_test_parameter": parameter_max,
    }


def lbfgs(parameters, inner: int) -> torch.optim.LBFGS:
    return torch.optim.LBFGS(
        parameters,
        lr=1.0,
        max_iter=inner,
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=1.0e-12,
        tolerance_change=1.0e-14,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("initial_model", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-seed", type=int, default=3102)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--quadrature-order", type=int, default=256)
    parser.add_argument("--audit-order", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--inner-u", type=int, default=5)
    parser.add_argument("--inner-v", type=int, default=5)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")
    if not args.initial_model.is_file():
        raise FileNotFoundError(args.initial_model)
    if args.audit_order <= args.quadrature_order:
        raise ValueError("audit_order must exceed quadrature_order")

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    trial = TrialNet(args.width, args.depth).to(args.device)
    initial_trial = torch.load(
        args.initial_model, map_location="cpu", weights_only=True
    )
    trial.load_state_dict(initial_trial)

    torch.manual_seed(args.test_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.test_seed)
    test = TrialNet(args.width, args.depth).to(args.device)

    train_rule = gauss_rule(args.quadrature_order, args.device)
    refined_rule = gauss_rule(args.audit_order, args.device)
    refined_basis = fourier_data(refined_rule[0], args.modes)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path("fourier_noninterface") / "runs" / f"wan_1d_paired_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "method": "learned_adversary_wan_logv_rawu",
        "seed": args.seed,
        "test_seed": args.test_seed,
        "modes_for_diagnostics": args.modes,
        "quadrature_order": args.quadrature_order,
        "audit_order": args.audit_order,
        "rounds": args.rounds,
        "inner_u": args.inner_u,
        "inner_v": args.inner_v,
        "width": args.width,
        "depth": args.depth,
        "device": args.device,
        "initial_model": str(args.initial_model.resolve()),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2))
    torch.save(trial.state_dict(), output / "initial_model.pt")
    torch.save(test.state_dict(), output / "initial_test.pt")

    calls_u = 0
    calls_v = 0
    accepted_round = 0
    stop_reason: str | None = None
    history: list[dict[str, float | int | bool | str | None]] = []
    start = time.perf_counter()

    print("=" * 104)
    print("PAIRED LEARNED-ADVERSARY WAN 1D POISSON + ALTERNATING L-BFGS")
    print("=" * 104)
    print("Trial initialization, PDE, hard boundary, and quadrature match the DFR comparison.")
    print("v maximizes log normalized weak score; u minimizes the raw score.")
    print("The Fourier N64 residual below is diagnostic only, not the WAN training objective.")
    print(f"Initial trial: {args.initial_model.resolve()}")
    print(
        f"Q={args.quadrature_order}/{args.audit_order} rounds={args.rounds} "
        f"inner_u/v={args.inner_u}/{args.inner_v} test_seed={args.test_seed}"
    )
    print(f"Output: {output.resolve()}\n", flush=True)

    def report(round_number: int, stage: str) -> dict[str, float | int | bool | str | None]:
        result = audit(trial, test, train_rule, refined_rule)
        metrics = diagnostics(trial, refined_rule, refined_basis)
        record: dict[str, float | int | bool | str | None] = {
            "round": round_number,
            "stage": stage,
            "calls_u": calls_u,
            "calls_v": calls_v,
            "seconds": time.perf_counter() - start,
            **result,
            **metrics,
        }
        history.append(record)
        print(
            f"round={round_number:4d} stage={stage:8s} calls_u/v={calls_u}/{calls_v} "
            f"weak_Q={result['score_q']:.6e} refined={result['score_refined']:.6e} "
            f"gap={result['relative_gap']:.3e} audit={'PASS' if result['passed'] else 'FAIL'}"
        )
        print(
            f"  absolute_L2={metrics['absolute_l2']:.6e} "
            f"relative_L2={metrics['relative_l2']:.6e} "
            f"H1={metrics['h1']:.6e} Fourier_N{args.modes}={metrics['fourier_loss']:.6e}"
        )
        print(
            f"  exact_test_abs={result['exact_normalized_abs']:.3e} "
            f"max_test_parameter={result['max_test_parameter']:.3e}",
            flush=True,
        )
        return record

    initial_record = report(0, "initial")
    if not initial_record["passed"]:
        raise RuntimeError("Initial learned test function failed the quadrature audit")

    for round_number in range(1, args.rounds + 1):
        trial_backup = copy.deepcopy(trial.state_dict())
        test_backup = copy.deepcopy(test.state_dict())

        set_trainable(trial, False)
        set_trainable(test, True)
        optimizer_v = lbfgs(test.parameters(), args.inner_v)

        def closure_v() -> torch.Tensor:
            nonlocal calls_v
            calls_v += 1
            optimizer_v.zero_grad(set_to_none=True)
            score = weak_quantities(trial, test, train_rule)["score"]
            loss = -torch.log(score + EPS)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite WAN test objective")
            loss.backward()
            return loss

        optimizer_v.step(closure_v)
        after_v = audit(trial, test, train_rule, refined_rule)
        if not after_v["passed"]:
            trial.load_state_dict(trial_backup)
            test.load_state_dict(test_backup)
            stop_reason = f"round {round_number}: after-v audit failed"
            print(f"STOP: {stop_reason}; both networks rolled back.", flush=True)
            break

        set_trainable(test, False)
        set_trainable(trial, True)
        optimizer_u = lbfgs(trial.parameters(), args.inner_u)

        def closure_u() -> torch.Tensor:
            nonlocal calls_u
            calls_u += 1
            optimizer_u.zero_grad(set_to_none=True)
            loss = weak_quantities(trial, test, train_rule)["score"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite WAN trial objective")
            loss.backward()
            return loss

        optimizer_u.step(closure_u)
        after_u = audit(trial, test, train_rule, refined_rule)
        if not after_u["passed"]:
            trial.load_state_dict(trial_backup)
            test.load_state_dict(test_backup)
            stop_reason = f"round {round_number}: after-u audit failed"
            print(f"STOP: {stop_reason}; both networks rolled back.", flush=True)
            break

        accepted_round = round_number
        if round_number == 1 or round_number % 10 == 0 or round_number == args.rounds:
            report(round_number, "accepted")

    set_trainable(trial, True)
    set_trainable(test, True)
    final_record = report(accepted_round, "final")
    state = {
        "trial": trial.state_dict(),
        "test": test.state_dict(),
        "config": config,
        "accepted_round": accepted_round,
        "calls_u": calls_u,
        "calls_v": calls_v,
        "metrics": final_record,
        "stop_reason": stop_reason,
        "status": "guarded_stop" if stop_reason else "bounded_pilot_finished",
    }
    torch.save(state, output / "final_checkpoint.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2))

    print("\n" + "=" * 104)
    print("PAIRED WAN FINISHED")
    print("=" * 104)
    print(f"accepted_round  : {accepted_round}")
    print(f"calls_u        : {calls_u}")
    print(f"calls_v        : {calls_v}")
    print(f"training_seconds: {time.perf_counter() - start:.6f}")
    print(f"stop_reason    : {stop_reason}")
    print(f"absolute_L2    : {float(final_record['absolute_l2']):.10e}")
    print(f"relative_L2    : {float(final_record['relative_l2']):.10e}")
    print(f"H1             : {float(final_record['h1']):.10e}")
    print("This is a paired smooth-problem baseline, not a superiority claim.")


if __name__ == "__main__":
    main()
