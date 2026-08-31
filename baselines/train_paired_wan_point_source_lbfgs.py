#!/usr/bin/env python3
"""Paired learned-adversary WAN for the 1D Dirac point-source problem.

Objective classification: learned-neural-test weak moment with exact point load.
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

from src.common.train_fourier_dfr_1d_lbfgs import TrialNet, derivative
from experiments.dirac1d.train_fourier_dfr_point_source_lbfgs import (
    SOURCE,
    basis_data,
    diagnostics,
    exact_solution,
    split_rule,
)
from baselines.train_paired_wan_1d_lbfgs import EPS, lbfgs, max_parameter, set_trainable


def weak_quantities(trial: TrialNet, test: TrialNet, rule):
    coordinates, weights = rule
    x = coordinates.detach().clone().requires_grad_(True)
    u = trial(x)
    v = test(x)
    du = derivative(u, x)
    dv = derivative(v, x)
    source_x = torch.full((1, 1), SOURCE, device=x.device, dtype=x.dtype)
    source_value = test(source_x)[0, 0]
    moment = (weights * du * dv).sum() - source_value
    norm2 = (weights * (v.square() + dv.square())).sum()
    score = moment.square() / (norm2 + EPS)

    _, exact_gradient = exact_solution(x)
    exact_moment = (weights * exact_gradient * dv).sum() - source_value
    exact_normalized_abs = exact_moment.abs() / torch.sqrt(norm2 + EPS)
    return {
        "score": score,
        "exact_normalized_abs": exact_normalized_abs,
    }


def audit(trial, test, train_rule, refined_rule):
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
        math.isfinite(v)
        for v in (score_q, score_refined, relative_gap, exact_abs, parameter_max)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("initial_model", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-seed", type=int, default=3102)
    parser.add_argument("--modes", type=int, default=256)
    parser.add_argument("--quadrature-order-per-half", type=int, default=1024)
    parser.add_argument("--audit-order-per-half", type=int, default=2048)
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
    if args.audit_order_per_half <= args.quadrature_order_per_half:
        raise ValueError("audit order must exceed training order")

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    trial = TrialNet(args.width, args.depth).to(args.device)
    trial.load_state_dict(
        torch.load(args.initial_model, map_location="cpu", weights_only=True)
    )
    torch.manual_seed(args.test_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.test_seed)
    test = TrialNet(args.width, args.depth).to(args.device)

    train_rule = split_rule(args.quadrature_order_per_half, args.device)
    refined_rule = split_rule(args.audit_order_per_half, args.device)
    refined_basis = basis_data(refined_rule[0], args.modes)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path("fourier_noninterface") / "runs" / f"wan_point_source_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "method": "learned_adversary_wan_point_source_logv_rawu",
        "seed": args.seed,
        "test_seed": args.test_seed,
        "modes_for_diagnostics": args.modes,
        "quadrature_order_per_half": args.quadrature_order_per_half,
        "audit_order_per_half": args.audit_order_per_half,
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
    stop_reason = None
    history = []
    start = time.perf_counter()

    print("=" * 106)
    print("PAIRED LEARNED-ADVERSARY WAN: 1D DIRAC POINT SOURCE")
    print("=" * 106)
    print("Weak form: integral u'v' - v(pi/2); exact solution is H1 but not H2")
    print("v maximizes log normalized weak score; u minimizes raw score")
    print(f"Initial trial: {args.initial_model.resolve()}")
    print(
        f"split_Q={args.quadrature_order_per_half}/{args.audit_order_per_half} "
        f"rounds={args.rounds} inner_u/v={args.inner_u}/{args.inner_v}"
    )
    print(f"Output: {output.resolve()}\n", flush=True)

    def report(round_number: int, stage: str):
        result = audit(trial, test, train_rule, refined_rule)
        metrics = diagnostics(trial, refined_rule, refined_basis)
        record = {
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
            f"relative_L2={metrics['relative_l2']:.6e} H1={metrics['h1']:.6e} "
            f"Fourier_N{args.modes}={metrics['fourier_loss']:.6e}"
        )
        print(
            f"  exact_test_abs={result['exact_normalized_abs']:.3e} "
            f"max_test_parameter={result['max_test_parameter']:.3e}",
            flush=True,
        )
        return record

    initial = report(0, "initial")
    if not initial["passed"]:
        raise RuntimeError("Initial point-source test failed audit")

    for round_number in range(1, args.rounds + 1):
        trial_backup = copy.deepcopy(trial.state_dict())
        test_backup = copy.deepcopy(test.state_dict())

        set_trainable(trial, False)
        set_trainable(test, True)
        optimizer_v = lbfgs(test.parameters(), args.inner_v)

        def closure_v():
            nonlocal calls_v
            calls_v += 1
            optimizer_v.zero_grad(set_to_none=True)
            score = weak_quantities(trial, test, train_rule)["score"]
            loss = -torch.log(score + EPS)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite point-source v objective")
            loss.backward()
            return loss

        optimizer_v.step(closure_v)
        if not audit(trial, test, train_rule, refined_rule)["passed"]:
            trial.load_state_dict(trial_backup)
            test.load_state_dict(test_backup)
            stop_reason = f"round {round_number}: after-v audit failed"
            print(f"STOP: {stop_reason}; both networks rolled back.", flush=True)
            break

        set_trainable(test, False)
        set_trainable(trial, True)
        optimizer_u = lbfgs(trial.parameters(), args.inner_u)

        def closure_u():
            nonlocal calls_u
            calls_u += 1
            optimizer_u.zero_grad(set_to_none=True)
            loss = weak_quantities(trial, test, train_rule)["score"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite point-source u objective")
            loss.backward()
            return loss

        optimizer_u.step(closure_u)
        if not audit(trial, test, train_rule, refined_rule)["passed"]:
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
    final = report(accepted_round, "final")
    torch.save(
        {
            "trial": trial.state_dict(),
            "test": test.state_dict(),
            "config": config,
            "accepted_round": accepted_round,
            "calls_u": calls_u,
            "calls_v": calls_v,
            "metrics": final,
            "stop_reason": stop_reason,
            "status": "guarded_stop" if stop_reason else "bounded_pilot_finished",
        },
        output / "final_checkpoint.pt",
    )
    (output / "history.json").write_text(json.dumps(history, indent=2))

    print("\n" + "=" * 106)
    print("POINT-SOURCE PAIRED WAN FINISHED")
    print("=" * 106)
    print(f"accepted_round  : {accepted_round}")
    print(f"calls_u         : {calls_u}")
    print(f"calls_v         : {calls_v}")
    print(f"training_seconds: {time.perf_counter() - start:.6f}")
    print(f"stop_reason     : {stop_reason}")
    print(f"absolute_L2     : {float(final['absolute_l2']):.10e}")
    print(f"relative_L2     : {float(final['relative_l2']):.10e}")
    print(f"H1              : {float(final['h1']):.10e}")
    print("This is a single-seed low-regularity baseline, not a superiority claim.")


if __name__ == "__main__":
    main()
