# Explicit weak Fourier-WAN self-check summary

This is a compact audited self-check record, not a training result.

- dtype/device: `torch.float64`, CPU read-only check.
- large-gradient1D exact weak residual: Q512/Q1024, **PASS**.
  - Q1024 maximum correct moment: `7.16937620381941088e-12`.
  - Q1024 correct weighted DFR score: `5.77604954645393597e-25`.
  - source-omitted DFR score: `1.53144024052603356e+02`.
  - source-sign-reversed DFR score: `6.12576096210412288e+02`.
- Poisson2D exact weak residual: Q96xQ96/Q192xQ192, **PASS**.
  - Q192 maximum correct moment: `1.14649973630467379e-12`.
  - Q192 correct weighted DFR score: `4.33320162752718210e-26`.
- frozen non-solution weak/strong equivalence: **PASS** for both problems.
- backward graph: finite, nonzero trial gradients and tensorwise unchanged parameters, **PASS**.
- AST structural check: `forbidden_call_nodes=[]`, **PASS**.
- Fourier basis trainable parameters: `0`.
- neural test networks: `0`.
- test optimizers: `0`.
- optimizer steps: `0`.
- parameter updates: `0`.
- checkpoints written: `0`.
- formal seed-2026 training: **pending**, because CUDA was unavailable during preflight.

Overall strict self-check decision: **PASS**.
