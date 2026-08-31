# Release manifest

This candidate was assembled from the non-Git research directory `wan-master/`. Package-import edits were made only in the candidate copies. The upstream project is <https://github.com/yaohua32/wan>.

Origin values are restricted to `upstream`, `modified_upstream`, `new_research_code`, `generated_documentation`, and `audited_result`.

| Published path | Original path | Category | Origin | Purpose | SHA-256 |
|---|---|---|---|---|---|
| `.gitignore` | `—` | GENERATED DOCUMENTATION | generated_documentation | Release documentation/configuration | `711923a42910b21140e8b5b0b674d219f213b1d433cff0a1843acca35dbb68c3` |
| `CITATION.cff` | `—` | GENERATED DOCUMENTATION | generated_documentation | Release documentation/configuration | `78166b18182e6672602be3e263c52f0b07017666e1add93e43a34c74d0048cf1` |
| `LICENSE` | `LICENSE` | UPSTREAM DEPENDENCIES | upstream | Upstream MIT license and copyright notice | `165b7023aa790b2b4cc69fdf0c791e84254fc853bef2ff044ae6fa465fab8952` |
| `README.md` | `—` | GENERATED DOCUMENTATION | generated_documentation | Release documentation/configuration | `7d4c1e417c2e8bd29a170e495031243f4ef49a41d7d613f65062519989764128` |
| `audits/__init__.py` | `—` | PACKAGE SUPPORT | generated_documentation | Python package marker | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `audits/audit_fourier_2d_diagonal_layer_training_grid.py` | `audit_fourier_2d_diagonal_layer_training_grid.py` | READ-ONLY AUDITS | new_research_code | Read-only spectral, endpoint, or grid audit | `af2d973c3a05208233761c3aa785a20dd2cbbc15c989c176319dac182b9cd410` |
| `audits/audit_fourier_dfr_spectral_tail.py` | `audit_fourier_dfr_spectral_tail.py` | READ-ONLY AUDITS | new_research_code | Read-only spectral, endpoint, or grid audit | `97d7a5f21f98da64c6fdf6bef5e84a63d91ffc3f33f141114d9c6b8e850f0238` |
| `audits/audit_fourier_point_source_spectral_tail.py` | `audit_fourier_point_source_spectral_tail.py` | READ-ONLY AUDITS | new_research_code | Read-only spectral, endpoint, or grid audit | `7f054d433073225388c38e1272d9f6e42a0e821164c34c7778c01c5b2854230d` |
| `baselines/__init__.py` | `—` | PACKAGE SUPPORT | generated_documentation | Python package marker | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `baselines/train_original_wan_2d_lbfgs.py` | `train_original_wan_2d_lbfgs.py` | BASELINES | new_research_code | Original WAN or PINN baseline entrypoint | `0b0a2d3c03207dfd3ff24124a5f11c5aac3f880e0c1b6213f5e7aed6b397f1e8` |
| `baselines/train_original_wan_large_gradient.py` | `train_original_wan_large_gradient.py` | BASELINES | new_research_code | Original WAN or PINN baseline entrypoint | `b71a1ce0eec7a5b49e6c6723fe42ae7ea608f3b67c9109c6e3746ec78475f12f` |
| `baselines/train_paired_pinn_1d_lbfgs.py` | `train_paired_pinn_1d_lbfgs.py` | BASELINES | new_research_code | Original WAN or PINN baseline entrypoint | `c0930a496481bad3b75c2f5b78b660c6c593b2188187f58e7cc85355c2f90236` |
| `baselines/train_paired_wan_1d_lbfgs.py` | `train_paired_wan_1d_lbfgs.py` | BASELINES | new_research_code | Original WAN or PINN baseline entrypoint | `f565ead5b48cad660742adaca7c800db156ac95bb971d6ac140cd3e301bd629c` |
| `baselines/train_paired_wan_point_source_lbfgs.py` | `train_paired_wan_point_source_lbfgs.py` | BASELINES | new_research_code | Original WAN or PINN baseline entrypoint | `2ad8273f10a384eda8755f10e7f07708ee2e2354b00b232aeb3865bcc960f37e` |
| `docs/FOURIER_WEAK_WAN_IMPLEMENTATION.md` | `FOURIER_WEAK_WAN_IMPLEMENTATION.md` | DOCUMENTATION | new_research_code | Research implementation design note | `ed84440ba4f15eb2fe16b63e9c53b7ea54e93f230f8ffe9d5c08b229ac19ea6b` |
| `environment.yml` | `—` | GENERATED DOCUMENTATION | generated_documentation | Release documentation/configuration | `626a4e4f6d9d52131b4f2d81e00a96bbea7525ca28f1016bfa7cfa49a351d1c8` |
| `experiments/__init__.py` | `—` | PACKAGE SUPPORT | generated_documentation | Python package marker | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `experiments/diagonal_layer2d/__init__.py` | `—` | PACKAGE SUPPORT | generated_documentation | Python package marker | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `experiments/dirac1d/__init__.py` | `—` | PACKAGE SUPPORT | generated_documentation | Python package marker | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `experiments/dirac1d/train_fourier_dfr_point_source_lbfgs.py` | `train_fourier_dfr_point_source_lbfgs.py` | TRAINING ENTRYPOINTS | new_research_code | Problem-specific or paired experiment entrypoint | `e42423885c1bb44cb66a5cfb4943d7285314d069d2a6a7b3440f21ca00674099` |
| `experiments/large_gradient1d/__init__.py` | `—` | PACKAGE SUPPORT | generated_documentation | Python package marker | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `experiments/large_gradient1d/train_fourier_large_gradient.py` | `train_fourier_large_gradient.py` | TRAINING ENTRYPOINTS | new_research_code | Problem-specific or paired experiment entrypoint | `1630fd46533895c3e03c47a329e93ba0534ce6859b32c19e2fd2d89c11bf1271` |
| `experiments/large_gradient1d/train_fourier_weak_wan_large_gradient.py` | `train_fourier_weak_wan_large_gradient.py` | TRAINING ENTRYPOINTS | new_research_code | Problem-specific or paired experiment entrypoint | `938bbc22671eae8f1e952dd517aa9b3cdf743000a6768ee957031e9258a03062` |
| `experiments/paired_multiseed/__init__.py` | `—` | PACKAGE SUPPORT | generated_documentation | Python package marker | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `experiments/paired_multiseed/run_paired_multiseed_benchmark.py` | `run_paired_multiseed_benchmark.py` | TRAINING ENTRYPOINTS | new_research_code | Problem-specific or paired experiment entrypoint | `c08a2c6d24f5ac4433ff5dbc12d1c34048f2cac54709b24942694303eec675ed` |
| `experiments/poisson2d/__init__.py` | `—` | PACKAGE SUPPORT | generated_documentation | Python package marker | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `experiments/poisson2d/train_fourier_dfr_2d_lbfgs.py` | `train_fourier_dfr_2d_lbfgs.py` | TRAINING ENTRYPOINTS | new_research_code | Problem-specific or paired experiment entrypoint | `fbfaef9a1dca71881cc39107232c3bbac5b19069d1157a09a9a45ae54ec23c5e` |
| `experiments/poisson2d/train_fourier_weak_wan_poisson2d.py` | `train_fourier_weak_wan_poisson2d.py` | TRAINING ENTRYPOINTS | new_research_code | Problem-specific or paired experiment entrypoint | `505906d41bda7847257e0bfcd00949367679badecbc98b43bd0bee86438c9936` |
| `experiments/poisson2d/train_spectral_wan.py` | `train_spectral_wan.py` | TRAINING ENTRYPOINTS | new_research_code | Problem-specific or paired experiment entrypoint | `f3adf22412f78662c15927d290b44df647a73ae5be6fccadbfcebeff40d73ce7` |
| `requirements.txt` | `—` | GENERATED DOCUMENTATION | generated_documentation | Release documentation/configuration | `d5d7e68bcd721f7b07ce4cc074b1c37daa9a09730df537617b53fabdc22f435c` |
| `results/RESULT_INTERPRETATION.md` | `—` | GENERATED DOCUMENTATION | generated_documentation | Release documentation/configuration | `cc0f2a5507999e5e1533735c6f24468f60f7b783820c84adeec9f1eea6b40c7c` |
| `results/fourier_2d_diagonal_layer_grid_audit.json` | `fourier_2d_diagonal_layer_grid_audit.json` | RESULT SUMMARIES | audited_result | Compact audited result or self-check evidence | `0055637970a3fb3393dd44b6b8cbd8cd97a21e7e97ce6b38dbcaecce4cdccf81` |
| `results/fourier_weak_wan_selfcheck_summary.md` | `—` | RESULT SUMMARIES | audited_result | Compact audited result or self-check evidence | `37638abb15c561c2c4fab34870539fb0ba606e8bcffa9335b2940f84ddb4a2fa` |
| `results/paired_multiseed_results.json` | `paired_multiseed_runs/paired_multiseed_results.json` | RESULT SUMMARIES | audited_result | Compact audited result or self-check evidence | `17e8bf469547ee38ba7e3d1b9dd81cdd6ff813b676fb06a86531ccd9eefb9e97` |
| `results/paired_multiseed_summary.md` | `paired_multiseed_runs/paired_multiseed_summary.md` | RESULT SUMMARIES | audited_result | Compact audited result or self-check evidence | `b35020ded7282a5c0dc32b6c45eea0634f5010da8fded369a6b4427f2480d4bb` |
| `selfchecks/__init__.py` | `—` | PACKAGE SUPPORT | generated_documentation | Python package marker | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `selfchecks/fourier_2d_diagonal_layer_selfcheck.py` | `fourier_2d_diagonal_layer_selfcheck.py` | MATHEMATICAL SELFCHECKS | new_research_code | Read-only mathematical and structural check | `4651394c17c3ceeaa34fc0f53121b038532a67fb1b86560c279d529f6dec5ce9` |
| `selfchecks/fourier_point_source_selfcheck.py` | `fourier_point_source_selfcheck.py` | MATHEMATICAL SELFCHECKS | new_research_code | Read-only mathematical and structural check | `6a98f8c50e95b6e02a0eb950eff8229c698b229260a979134914045a38a144d5` |
| `selfchecks/fourier_test_space_selfcheck.py` | `fourier_test_space_selfcheck.py` | MATHEMATICAL SELFCHECKS | new_research_code | Read-only mathematical and structural check | `fb7d4077f1837a2243bc1ae7d09b1c35b63ba2d4bdd49f43fe75e3f967f21535` |
| `selfchecks/selfcheck_fourier_weak_wan_loss.py` | `selfcheck_fourier_weak_wan_loss.py` | MATHEMATICAL SELFCHECKS | new_research_code | Read-only mathematical and structural check | `37028daba448eeb1057e2e01d33cd95c9e49a17b4eeb3a46998531a2fe1cbf75` |
| `src/__init__.py` | `—` | PACKAGE SUPPORT | generated_documentation | Python package marker | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/common/__init__.py` | `—` | PACKAGE SUPPORT | generated_documentation | Python package marker | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/common/train_fourier_dfr_1d_lbfgs.py` | `train_fourier_dfr_1d_lbfgs.py` | CORE METHOD | new_research_code | Shared trial network and numerical helpers | `288e03631d0be92e26c129b9b92774e6761ef446b610871ff650af331db292ec` |
| `src/methods/__init__.py` | `—` | PACKAGE SUPPORT | generated_documentation | Python package marker | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/methods/fourier_test_space.py` | `fourier_test_space.py` | CORE METHOD | new_research_code | Fourier test-space or DFR loss implementation | `779983b5a1fc7d793010b9f91c9a934e34ec39a076720219df4e515bcbc4ba90` |
| `src/methods/fourier_weak_wan_loss.py` | `fourier_weak_wan_loss.py` | CORE METHOD | new_research_code | Fourier test-space or DFR loss implementation | `3145fbc8c54d132355225f4bc84e66893974edc0efd0702bd665ae6276c5ba87` |
| `src/methods/wan_spectral_loss.py` | `wan_spectral_loss.py` | CORE METHOD | new_research_code | Fourier test-space or DFR loss implementation | `5264611c0ff0575daee2b52b5cd8fd56aebce0d29ff469359a559f2705bcf110` |
| `src/problems/__init__.py` | `—` | PACKAGE SUPPORT | generated_documentation | Python package marker | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `tests/__init__.py` | `—` | PACKAGE SUPPORT | generated_documentation | Python package marker | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `tests/test_fourier_space.py` | `test_fourier_space.py` | TESTS | new_research_code | Lightweight unit test | `50a486f20d3780db5aae26deb639f664b95470822d9e9b40753980a001d883f1` |

`MANIFEST.md` intentionally omits its own SHA-256 because embedding it is self-referential.

## Important exclusions

- No checkpoints, model parameters, logs, full `runs/`, artifacts, caches, or temporary files are included.
- No explicit-weak training result is included or claimed; CUDA training remains pending.
- The requested historical name `audit_fourier_dfr_point_source_tail.py` was absent; the real `audit_fourier_point_source_spectral_tail.py` is included.
- Legacy/interface experiments and non-authoritative single-run outputs remain excluded.
