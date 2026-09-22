# WAN with Fourier Test Spaces and DFR Evaluation

This candidate research repository separates **Original WAN (learned neural test)**, **Fourier-strong DFR**, and **Fourier-test WAN (explicit weak moments)** for elliptic problems. It is prepared for source review and reproducibility inspection; it contains no checkpoints or full run directories.

## Poisson–Boltzmann DFR: LM Implementation

This repository now includes a review snapshot of the Poisson–Boltzmann interface Fourier weak-residual neural solver. It uses singularity splitting and a hard-boundary ansatz, with an `Adam3000 → direct LM` workflow and full-H1 DFR weights `1 + lambda`. The snapshot is provided for code and computational-cost review; it does not claim that staged performance measurements have identified a single bottleneck.

- [English LM code review](reviews/pb_dfr_lm_w32_k64/LM_CODE_REVIEW.md)
- [Residual and Jacobian implementation](reviews/pb_dfr_lm_w32_k64/source/parameter_study/W32_K64_DIRECT/numerics.py)
- [Direct LM training loop and checkpoint handling](reviews/pb_dfr_lm_w32_k64/source/parameter_study/W32_K64_DIRECT/background.py)
- [W32/K64 frozen configuration](reviews/pb_dfr_lm_w32_k64/source/parameter_study/W32_K64_DIRECT/frozen_config.json)
- [Endpoint results and verification](reviews/pb_dfr_lm_w32_k64/evidence/postprocess.json)
- [Source and checkpoint provenance](reviews/pb_dfr_lm_w32_k64/evidence/RUN_RECORD.md)

## Research question

Does replacing the learned adversarial test network in WAN with a fixed, normalized Fourier test space produce a more accurate and efficient finite-dimensional residual objective, and are the observed gains robust beyond solutions aligned with a few low Fourier modes?

## Methods and mathematical formulation

Original WAN (learned neural test) starts from the variational inf–sup problem

\[
\inf_u\sup_{v_\phi}\frac{|b(u,v_\phi)-F(v_\phi)|}{\|v_\phi\|_V},
\]

and alternates optimization of a trial network and a neural test network.

Both Fourier formulations fix

\[
V_N=\operatorname{span}\{\varphi_k\}_{k=1}^N
\]

and evaluates the squared restricted dual norm

\[
\sum_k\frac{|b(u,\varphi_k)-F(\varphi_k)|^2}{\lambda_k}.
\]

DFR is **not** a test function. The test functions are normalized Fourier sine functions; DFR evaluates the residual norm over the complete truncated Fourier test space.

There are three strictly distinguished methods in this snapshot:

- **Original WAN (learned neural test)** directly evaluates the weak pairing while alternately training trial and neural test networks.
- **Fourier-strong DFR** projects the strong residual `-Delta(u)-f`, uses second spatial derivatives, and has no neural test network. The currently published three-seed `large_gradient1d` and `poisson2d` Fourier results use this formulation.
- **Fourier-test WAN (explicit weak moments)** directly computes `b(u,phi)-F(phi)` for fixed Fourier sine tests, uses only first spatial derivatives, and applies DFR weights to the resulting modal residuals. It has neither a neural test network nor a test optimizer. The existing Dirac1D Fourier branch already uses explicit weak moments, including its point load.

The currently published three-seed large-gradient1D and Poisson2D Fourier results use the strong-residual spectral formulation. They must not be presented as results of the new explicit weak-moment implementation.

For Dirichlet axes on `(0, pi)`, the continuous normalized basis is `sqrt(2/pi) sin(kx)` in 1D and `(2/pi) sin(kx)sin(ly)` in 2D. The full H1 test norm gives weights `1/(1+k^2)` and `1/(1+k^2+l^2)`.

## Repository structure

- `src/methods/`: Fourier/DST test-space and spectral dual-loss implementations.
- `src/common/`: shared 1D networks and numerical helpers.
- `experiments/`: paired benchmark and problem-specific Fourier-test entrypoints.
- `baselines/`: Original WAN and PINN comparison entrypoints.
- `selfchecks/`: mathematical weak-form and forcing checks; no training.
- `audits/`: read-only endpoint, spectral-tail, and training-grid audits.
- `results/`: compact audited summaries only.
- `tests/`: lightweight Fourier-space tests.

Run commands below from the repository root with `PYTHONPATH=.`.

## Environment and installation

The tested environment used Python 3.12, PyTorch 2.5.1, NumPy, Matplotlib, and a PyTorch CUDA 11.8 build. CPU is supported for read-only mathematical checks. The formal paired benchmark was designed for CUDA.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH=.
```

For a conda environment:

```bash
conda env create -f environment.yml
conda activate wan-dfr-fourier
export PYTHONPATH=.
```

## Reproduction commands

Examples of method-specific entrypoints:

```bash
PYTHONPATH=. python src/common/train_fourier_dfr_1d_lbfgs.py --help
PYTHONPATH=. python experiments/dirac1d/train_fourier_dfr_point_source_lbfgs.py --help
PYTHONPATH=. python experiments/large_gradient1d/train_fourier_weak_wan_large_gradient.py --help
PYTHONPATH=. python experiments/poisson2d/train_fourier_weak_wan_poisson2d.py --help
PYTHONPATH=. python baselines/train_paired_wan_1d_lbfgs.py --help
PYTHONPATH=. python baselines/train_paired_pinn_1d_lbfgs.py --help
```

These are training programs; inspect configurations before running them. No training is required for the checks below.

### Mathematical self-checks

```bash
PYTHONPATH=. python selfchecks/fourier_test_space_selfcheck.py
PYTHONPATH=. python selfchecks/fourier_point_source_selfcheck.py
PYTHONPATH=. python -m selfchecks.selfcheck_fourier_weak_wan_loss \
  --device cpu --large-order 512 --large-refined-order 1024
PYTHONPATH=. python selfchecks/fourier_2d_diagonal_layer_selfcheck.py \
  --device cpu --beta 20 --modes 32 \
  --quadrature-order 384 --refined-order 768 --mode-chunk-size 128
```

### Strict paired three-seed benchmark

```bash
PYTHONPATH=. python experiments/paired_multiseed/run_paired_multiseed_benchmark.py \
  --device cuda --output-dir paired_multiseed_runs
```

The configured seeds are `[42, 2026, 3407]`. The published summary uses every seed and final endpoint: no best-seed selection and no best-checkpoint selection.

### Read-only audits

Endpoint re-audit after a benchmark run:

```bash
PYTHONPATH=. python experiments/paired_multiseed/run_paired_multiseed_benchmark.py \
  --device cuda --output-dir paired_multiseed_runs --reaudit-only
```

Diagonal-layer training-grid audit without training:

```bash
PYTHONPATH=. python audits/audit_fourier_2d_diagonal_layer_training_grid.py \
  --device cpu --beta 20 \
  --grid-sizes 64 128 256 384 512 \
  --mode-counts 8 16 32 64 \
  --reference-order 384 --refined-reference-order 768 \
  --output-json diagonal_layer_grid_audit.json
```

## Main strictly paired results

Mean ± sample standard deviation (`ddof=1`), seeds `[42, 2026, 3407]`:

| Problem | Method | Relative L2 | H1 | Wall time (s) |
|---|---|---:|---:|---:|
| dirac1d | Fourier-test WAN (DFR evaluation) | 1.8807e-2 ± 6.0742e-3 | 1.3882e-1 ± 3.2525e-2 | 2.6891 ± 0.0492 |
| dirac1d | Original WAN (learned neural test) | 2.7138e-1 ± 1.9228e-1 | 5.2818e-1 ± 1.3897e-1 | 41.5192 ± 0.2422 |
| large_gradient1d | Fourier-strong DFR | 1.0474e-1 ± 7.3712e-2 | 5.1366e-1 ± 3.5360e-1 | 3.3209 ± 0.0975 |
| large_gradient1d | Original WAN (learned neural test) | 5.2886e-1 ± 1.4305e-1 | 1.2085e1 ± 5.2026e-1 | 29.5920 ± 2.6740 |
| poisson2d | Fourier-strong DFR | 7.7505e-4 ± 1.1195e-4 | 6.1601e-3 ± 1.0162e-3 | 4.1882 ± 0.1957 |
| poisson2d | Original WAN (learned neural test) | 1.0426 ± 2.7505e-2 | 2.8729 ± 6.0511e-2 | 37.7517 ± 2.6308 |

All 18 endpoints passed their independent authoritative audit (one endpoint is marked `PASS_REAUDITED`). See `results/paired_multiseed_summary.md` and the path-sanitized `results/paired_multiseed_results.json`.

The explicit weak Fourier-WAN implementation has passed exact-solution, source-sign, quadrature-refinement, frozen weak/strong equivalence, backward-graph, and AST structural checks. Formal seed-2026 training is pending because CUDA was unavailable during the preflight. No explicit-weak training error or timing result is claimed here.

## Diagonal-layer status and limitations

**Mathematical and quadrature self-check completed; training not yet reported.** The smooth manufactured solution has a sharp nonseparable layer along `x+y=pi`. Q384/Q768 continuous weak-form checks converge, but the current training-grid audit found no passing `(grid, N)` combination among grids 64–512 and modes 8–64 under the fixed criteria. In particular, `grid-size=64` has only 62 interior points and cannot represent 64 independent DST modes.

The three-problem benchmark is small, uses three seeds, and is not evidence of general superiority. The present 2D spectral trainer projects a strong residual with a DST-II matrix whose assumed cell-center locations differ from the actual endpoint-excluded uniform samples. A pointwise-small strong residual cannot replace independent convergence checks of the two weak-form terms.

## Upstream attribution, citation, and license

This work is based on or informed by the WAN repository at <https://github.com/yaohua32/wan>. The retained MIT license is the upstream license and copyright notice (`Copyright (c) 2020 yaohua32`). `MANIFEST.md` distinguishes upstream, modified-upstream, newly written research code, generated documentation, and audited results.

Use `CITATION.cff` as the repository-level citation metadata. No paper title, DOI, or individual authorship is asserted because those details could not be confirmed from the local files. See `LICENSE` for redistribution terms.
