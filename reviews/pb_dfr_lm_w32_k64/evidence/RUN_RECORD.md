# Run record and source limitations

The endpoint evidence comes from `W32_K64_DIRECT/G-LM/final_checkpoint.pt`, SHA256 `ae605b519e7f26dac131bb62fe31cbba9c159d4e95414e3c8a3de3cb040136a1`. The checkpoint was not uploaded. `postprocess.json` points to that same endpoint and records the endpoint metrics and integration checks.

The checkpoint does not embed a Git commit or source-file hashes. The copied source hashes in `source_hashes.json` therefore document the review snapshot, not a byte-for-byte proof of the historical process image. The public frozen config replaces the private absolute initial-checkpoint path with a configurable token; the original path is intentionally omitted from this public snapshot.

The final status snapshot had known bookkeeping limitations: it mixed final accepted-step count with stale DFR/rejection fields from the last periodic status write; cumulative elapsed time held Adam-only time; optimization_seconds was zero; and per-candidate proposal rows were not saved. These limitations are retained and explained in `../LM_CODE_REVIEW.md`.
