# P9-H8 continual-learning method study

This is a separate method study. It does not alter the frozen P0–P12 protocol
results or the existing C3 runs. The machine-readable contract is
`configs/p9_h8_cl_method_study.json`.

## Shared setting

- DDI2025 train/validation/test parquet splits; no test data for selection.
- P9 task schedule per seed: `[38, 20, 20, 20, 20, 20, 20, 20]` classes.
- T-DDI backbone, focal gamma 1, AdamW, learning rate 0.001, weight decay 0.0001.
- Offline methods: up to 20 epochs/task, patience 5, validation seen-class Macro-F1.
- Fixed replay methods: 6,800 retained exemplars and 6,800 old-sample draws per
  epoch; current-task examples appear once per epoch. C0 and C6 use no buffer.

## Method implementations

| ID | Runner method | Track | Project-specific behavior |
|---|---|---|---|
| C0 | `sequential` | offline | Current task only; no replay. |
| C1 | `er_fixed_budget_uniform` | offline | Same buffer, exemplar ranking and sampler as C3; no teacher. |
| C2 | `derpp_fixed_budget_uniform` | offline | Current-label loss, replay-label loss and write-time logit MSE. Historical logits are keyed by raw class ID across head expansion. |
| C3 | `replay_distill_fixed_budget_uniform` | offline | Existing five-seed baseline; not rerun. |
| C4 | `xder_fixed_budget_uniform` | offline | C2 plus future-past logit refresh and old/new head margin constraints. An expanding head has no outputs for unseen classes, so the original X-DER future-head regularizer is inapplicable. |
| C6 | `perturb_merge_full_network` | offline | Stochastic task-vector perturbation and diagonal-Fisher closed-form merge coefficient. This uses full T-DDI weights instead of the original LoRA task vectors; newly introduced head rows stay at their trained values. |
| C0O/C1O | `sequential`/`er_fixed_budget_uniform` | single-pass | One pass per task; controls for C5. |
| C5 | `otc_mmot_online_adapted` | single-pass | Dynamic diagonal-Gaussian centroids, balanced entropic assignment, preservation loss, centroid-guided exemplar choice and Mahalanobis inference. P9 task boundaries remain available for evaluation and buffer reallocation, so this is an adaptation rather than a reproduction of task-free OTC. |

C7 DEMD is reserved for a later task-free dynamic-stream study. C5 is compared
with C0O/C1O; it is not ranked against 20-epoch C3 as an equal-compute run.
All results retain the original task-level and classwise audit artifacts.

Primary method references: [DER++](https://arxiv.org/abs/2004.07211),
[X-DER](https://arxiv.org/abs/2201.00766),
[OTC/MMOT](https://openaccess.thecvf.com/content/CVPR2026/html/Tran_An_Optimal_Transport-driven_Approach_for_Cultivating_Latent_Space_in_Online_CVPR_2026_paper.html),
[P&M](https://arxiv.org/abs/2505.22389), and
[DEMD](https://openaccess.thecvf.com/content/CVPR2025/html/Ye_Online_Task-Free_Continual_Learning_via_Dynamic_Expansionable_Memory_Distribution_CVPR_2025_paper.html).

## Run and inspect

```bash
tmux new-session -d -s p9_h8_cl -c "$PWD" \
  'bash -o pipefail -c "bash scripts/run_p9_cl_study.sh 2>&1 | tee outputs/runs_p9_cl_methods/driver.log"'
tail -f outputs/runs_p9_cl_methods/driver.log
.venv/bin/python scripts/analyze_p9_cl_methods.py
```

The runner checks RAM, disk, GPU memory, complete C3 baseline artifacts and
the source-code fingerprint. It skips complete runs and refuses to overwrite
partial runs. Results and paired deltas are written to
`outputs/runs_p9_cl_methods/analysis/`.

## Results as of 2026-09-16

All listed methods have five completed seeds (0–4). Values are final-task
Macro-F1, mean ± sample standard deviation, from
`outputs/runs_p9_cl_methods/analysis/method_summary.csv`.

| Track | Method | Macro-F1 |
|---|---|---:|
| Offline | C0 | 0.0359 ± 0.0121 |
| Offline | C1 | 0.3650 ± 0.0187 |
| Offline | C2 | 0.3565 ± 0.0192 |
| Offline | C3 | **0.3877 ± 0.0126** |
| Offline | C4 | 0.3723 ± 0.0126 |
| Offline | C6 | 0.0782 ± 0.0298 |
| Single-pass | C0O | 0.0072 ± 0.0026 |
| Single-pass | C1O | **0.2286 ± 0.0090** |
| Single-pass | C5 | 0.1637 ± 0.0218 |

C3 leads offline Macro-F1, while C4 has higher balanced accuracy
(`0.5969 ± 0.0353` versus C3 `0.4870 ± 0.0201`) and lower task forgetting
(`0.2324 ± 0.0349` versus `0.3489 ± 0.0262`). C5 trails its equal-pass C1O
control on Macro-F1. These are separate compute tracks. C4+GroupPrior is a
subsequent classifier correction study; its results are in
`docs/C4_GROUPPRIOR_PROGRESS.md`.
