# PB DFR direct LM implementation review: W32/K64

This directory is a review snapshot of the direct LM runner used for the PB DFR `W32_K64_DIRECT` run. It is intended to let a reviewer inspect the residual, Jacobian, damped solve, acceptance/retry path, checkpointing, and endpoint evidence without downloading model checkpoints.

## Source provenance and limits

The run directory was `<server-project>/parameter_study/W32_K64_DIRECT/`. The runtime checkpoint does not contain a source commit or source-file hash, so this snapshot cannot prove byte-for-byte identity with every file loaded by the historical process. `evidence/source_hashes.json` records hashes of the files copied from the current PB workspace. The numerical endpoint and checkpoint hash are preserved in `evidence/`; the source-identity limitation is intentional.

The public copy keeps the original runner behavior. It does not silently repair the following observed record limitations:

- final status writing did not refresh `current_DFR64` and `rejected_count` after the last 50-step status update;
- `cumulative_elapsed_seconds` retained the Adam-stage value and did not add LM wall time;
- `optimization_seconds` remained zero and is not a usable timing breakdown;
- `history.proposals` was empty, so per-candidate timing and the final retry sequence cannot be reconstructed.

No checkpoint is included. No new training or optimizer experiment was run for this review snapshot.

## Method

The trial is `u = u_s + u_r`; only the regular component is represented by the global neural trial. The network is `2-32-32-32-1`, tanh, float64, seed 2026, with the existing hard-boundary ansatz. The weak Fourier residual uses the full-H1 weight

`rho_kl = R_kl / sqrt(1 + lambda_kl)`

and `DFR64 = sum(rho_kl^2)`, with LM objective `F = 0.5 * DFR64`. The direct LM step actually used is

`(J J^T + mu I) y = rho`, `delta = -J^T y`.

No matrix inverse is formed. A candidate is accepted only when finite and when `F_candidate < F_current`. The direct-run damping divisor is 2; all other values are in [`source/parameter_study/W32_K64_DIRECT/frozen_config.json`](source/parameter_study/W32_K64_DIRECT/frozen_config.json).

## Dimensions and implementation locations

- width 32, parameter count `p = 2241`;
- K=64, residual dimension `m = 4096`;
- actual Jacobian shape `(4096, 2241)`;
- actual dual matrix shape `(4096, 4096)`.

The core review path is:

- residual evaluation and full-H1 weighting: [`numerics.py`](source/parameter_study/W32_K64_DIRECT/numerics.py), `evaluate()` and `residual_jacobian()`;
- Fourier basis, quadrature, `1+lambda` spectral weights and weak moments: [`pb_fourier_residual.py`](source/scripts/pb_fourier_residual.py), `basis()`, `quadrature()`, `moments()`, `metrics()`;
- PB splitting, exact solution, dielectric/source data and hard-boundary trial: [`pb_benchmark.py`](source/scripts/pb_benchmark.py), [`train_pb_dfr_smoke.py`](source/scripts/train_pb_dfr_smoke.py), and the shared solver modules under `source/full_solver/` and `source/width64_optimizer_study/`;
- parameter flattening and restoration: `vector()` and `assign()` in [`numerics.py`](source/parameter_study/W32_K64_DIRECT/numerics.py);
- Jacobian construction: `residual_jacobian()` uses batched `torch.autograd.grad`, eight residual rows per call, `retain_graph=True`, then concatenates blocks;
- Gram assembly and solve: [`background.py`](source/parameter_study/W32_K64_DIRECT/background.py), `phase_lm()`, `A = J @ J.T`, and `torch.linalg.solve(A + mu*I, rho)`;
- candidate evaluation, rejection restoration and damping retry: `phase_lm()`. The candidate is assigned with `assign(theta + delta)`. On rejection, `assign(theta)` restores the prior accepted weights and doubles `mu`; acceptance halves `mu`;
- accepted-step Jacobian refresh: after each accepted step, `residual_jacobian()` and `J @ J.T` are rebuilt;
- Adam→LM entry, stopping, checkpoint and resume: [`background.py`](source/parameter_study/W32_K64_DIRECT/background.py), `phase_adam()`, `phase_lm()`, `save()`, `load()`, and `main()`;
- endpoint-only evaluation: [`postprocess_only.py`](source/parameter_study/W32_K64_DIRECT/postprocess_only.py).

The Jacobian is not produced by `jacrev` or `jacfwd`; it is assembled by repeated batched reverse-mode `torch.autograd.grad`. The code creates a 4096-row identity on the GPU and retains the residual graph while processing blocks of eight rows. `numpy_functions()` moves diagnostic values and gradients to CPU NumPy arrays. Diagnostics run every 100 Adam steps and every 100 accepted LM steps; checkpoints are saved every 100 Adam steps and every 50 accepted LM steps, plus stage boundaries. The code calls `torch.cuda.synchronize()` around selected numerical operations. The evidence does not provide a separate timing for Jacobian, Gram assembly, or solve, so the performance bottleneck cannot be identified from this run alone. The 4096x4096 dual system is mathematically valid even though a 2241x2241 primal form is possible; primal was not used or numerically substituted here.

## Observed W32/K64 run

The run was `Adam3000 -> direct LM`; there was no L-BFGS stage. The final checkpoint records:

| quantity | value |
|---|---:|
| accepted LM steps | 179 |
| rejected attempts | 171 |
| attempts | 350 |
| absolute L2 | 7.451472889134215e-05 |
| absolute H1 | 4.377476094342059e-04 |
| DFR64 (n=128) | 4.15677274492407e-06 |
| DFR64 (n=96) | 4.1567727496018905e-06 |
| LM wall seconds in final checkpoint | 14462.573022493161 |
| configured LM cap | 14400 s |
| stop reason | `capped_not_converged` |
| cap type | `time_cap` |

The wall-time/accepted-step quotient is about 80.8 seconds per accepted step. It includes rejected attempts, Jacobian rebuilds, diagnostics, synchronization and checkpoint work; it is not a per-solve timing.

Final checkpoint SHA256: `ae605b519e7f26dac131bb62fe31cbba9c159d4e95414e3c8a3de3cb040136a1`.

## Verification and limitations

The endpoint was reloaded from the final checkpoint. L2 and H1 n96/n128 relative differences were `1.9278978799454612e-14` and `2.7244520907276543e-15`. DFR64 n96/n128 relative difference was `1.1253491719235805e-09`, below the recorded `1e-8` threshold. These are endpoint verification results, not evidence that the capped run converged.

The final checkpoint does not contain per-candidate proposal rows, and the recorded `optimization_seconds` is zero. Therefore this evidence cannot decompose the 4-hour wall time into residual, Jacobian, Gram, solve, transfer, synchronization, diagnostics, or checkpoint components. No specific performance bottleneck is claimed.

## Inspecting and running

The source tree is a review snapshot, not a drop-in replacement for the original absolute paths. Set `PYTHONPATH` to the copied `source/` tree and use a configurable output directory when adapting it elsewhere. The original run used Python `<validated-python>`, `CUDA_VISIBLE_DEVICES=0`, `cuda:0`, and float64; CUDA failure raises rather than falling back to CPU.

Inspection-only commands (from the repository root):

```bash
python -m py_compile reviews/pb_dfr_lm_w32_k64/source/parameter_study/W32_K64_DIRECT/background.py reviews/pb_dfr_lm_w32_k64/source/parameter_study/W32_K64_DIRECT/numerics.py reviews/pb_dfr_lm_w32_k64/source/parameter_study/W32_K64_DIRECT/postprocess_only.py
PYTHONPATH=reviews/pb_dfr_lm_w32_k64/source/parameter_study/W32_K64_DIRECT:reviews/pb_dfr_lm_w32_k64/source/width64_optimizer_study:reviews/pb_dfr_lm_w32_k64/source/capacity_study:reviews/pb_dfr_lm_w32_k64/source/fourier_resolution_study:reviews/pb_dfr_lm_w32_k64/source/full_solver/scripts:reviews/pb_dfr_lm_w32_k64/source/scripts python -c "import numerics; print(\"review import OK\")"
```

The copied `background.py` is the training entry and must not be run merely to inspect the code. `postprocess_only.py` is endpoint-only and requires a separately supplied compatible checkpoint; no checkpoint is included in this repository branch.
