# Explicit weak Fourier-test WAN implementation

The fixed test functions are normalized Fourier sine functions. DFR is not a
test function: it evaluates the squared residual dual norm over the truncated
Fourier test space. `fourier_weak_wan_loss.py` computes explicit weak moments
using only first spatial derivatives of the trial network.

The existing strong-residual spectral implementations remain unchanged as an
independent ablation. A later paired-driver revision should expose three
separate method identifiers:

1. `original_wan`: learned neural test and explicit WAN weak pairing;
2. `fourier_strong_dfr`: the preserved strong-residual DST/Fourier projection;
3. `fourier_weak_wan`: the new explicit weak Fourier moments with DFR weights.

The future driver should clone one trial initialization per problem/seed, load
the identical state into all three methods, preserve method-specific objective
labels in configs and results, and run independent endpoint quadrature audits.
No paired driver or existing result is changed in this stage.
