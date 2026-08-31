# Interpretation of published results

The numerical values in `paired_multiseed_summary.md` and
`paired_multiseed_results.json` are preserved from the strict paired run. Their
method identifier `dfr_in_wan` is historical and does not imply that every
problem used an explicit weak moment.

| Problem | Historical result method | Correct interpretation |
|---|---|---|
| `dirac1d` | `dfr_in_wan` | **Fourier-test WAN (DFR evaluation)**: explicit weak moments with exact point load |
| `large_gradient1d` | `dfr_in_wan` | **Fourier-strong DFR**: strong residual spectral projection |
| `poisson2d` | `dfr_in_wan` | **Fourier-strong DFR**: strong residual DST-II projection |
| all | `original_wan` | **Original WAN (learned neural test)** |

The currently published three-seed large-gradient1D and Poisson2D Fourier
results use the strong-residual spectral formulation. They must not be
presented as results of the new explicit weak-moment implementation.

The new explicit weak implementation has mathematical and code self-check
evidence only. Formal training was not run because the CUDA preflight failed.
No weak-method performance or timing values are reported.
