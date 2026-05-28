# HW3 LeRobot ACT Cross-Environment Generalization

This repository contains the code, report source, and result tables for **Task 2: LeRobot ACT cross-environment generalization on CALVIN**.

The experiment compares two ACT policies:

- `ACT-A`: trained only on CALVIN environment A.
- `ACT-ABC`: trained on mixed CALVIN environments A, B, and C with the same network architecture and hyperparameters.

Both policies are evaluated zero-shot on the unseen CALVIN environment D using offline Action L1 error. The assignment allows reporting either Success Rate or action error; this project uses action error because the full interactive CALVIN rollout environment was not required for the submitted experiment.

## Repository Structure

```text
.
├── configs/
│   └── act_calvin_common.env
├── report/
│   ├── final_report.tex
│   ├── figures/
│   └── tables/
├── scripts/
│   ├── prepare_env.sh
│   ├── split_calvin_lerobot_dataset.py
│   ├── verify_formal_calvin_data.py
│   ├── offline_eval_act_loss.py
│   ├── plot_act_results.py
│   ├── log_swanlab_from_csv.py
│   ├── train_act_a_clean.sh
│   ├── train_act_abc_clean.sh
│   └── eval_dd_l1_curve.sh
├── environment.yml
└── README.md
```

Large datasets and model weights are intentionally excluded from git.

## Environment

The completed experiments used:

- Python 3.10.20
- PyTorch 2.5.1 + CUDA 12.1
- LeRobot v0.3.3
- NVIDIA GeForce RTX 3060 12GB

Create an environment:

```bash
conda env create -f environment.yml
conda activate dl_hw3_lerobot_act
```

For the original machine layout, edit or source:

```bash
source configs/act_calvin_common.env
```

The project used LeRobot v0.3.3 because the CALVIN LeRobot datasets are in v2.1 format. Newer LeRobot versions use newer dataset formats and may require migration.

## Data Preparation

Download the LeRobot-format CALVIN datasets from Hugging Face:

```text
fywang/calvin-task-ABC-D-lerobot
fywang/calvin-task-D-D-lerobot
```

Important: `fywang/calvin-task-ABC-D-lerobot` contains official D validation episodes at the end, so it must not be used directly as the clean A+B+C training set.

The clean splits used in this project are:

```text
local/calvin-task-A-train-lerobot
local/calvin-task-ABC-train-lerobot
fywang/calvin-task-D-D-lerobot
```

The split script is:

```bash
python scripts/split_calvin_lerobot_dataset.py --help
```

After preparing the data, verify that LeRobot can read the datasets:

```bash
PYTHONPATH=$LEROBOT_REPO/src python scripts/verify_formal_calvin_data.py
```

## Training

Train ACT on environment A only:

```bash
bash scripts/train_act_a_clean.sh
```

Train ACT on environments A+B+C:

```bash
bash scripts/train_act_abc_clean.sh
```

Both scripts use:

- batch size: 64
- training steps: 200000
- learning rate: 1e-5
- optimizer: AdamW
- ACT chunk size: 10
- save frequency: 5000 steps

## Testing / Evaluation

Evaluate both trained policies on unseen environment D with offline Action L1:

```bash
bash scripts/eval_dd_l1_curve.sh
```

This evaluates checkpoints:

```text
025000, 050000, 075000, 100000, 125000, 150000, 175000, 200000
```

Each checkpoint is evaluated on 200 batches with batch size 64, for 12800 samples.

## Plotting

Regenerate the training and validation curves:

```bash
python scripts/plot_act_results.py
```

Replay the parsed scalar metrics into local SwanLab experiments and export dashboard figures:

```bash
python scripts/log_swanlab_from_csv.py
swanlab watch swanlab_logs
```

The report figures named `swanlab_*.png` were exported from the SwanLab dashboard. The CSV files remain in `report/tables/` so the curves can be replayed exactly.

Report figures exported from SwanLab:

- `report/figures/swanlab_train_loss_after_5k.png`
- `report/figures/swanlab_dd_l1_mean.png`
- `report/figures/swanlab_dashboard_curves.png`

Optional local diagnostic figures from `plot_act_results.py`:

- `report/figures/act_train_and_dd_eval.png`
- `report/figures/act_train_loss_full_log.png`
- `report/figures/act_train_loss_after_5k.png`
- `report/figures/act_dd_l1_curve.png`

## Main Results

Final 200K checkpoint comparison:

| Model | Training data | Final train loss | D-D Action L1 |
| --- | --- | ---: | ---: |
| ACT-A | A | 0.164 | 0.2990 |
| ACT-ABC | A+B+C | 0.228 | 0.2757 |

Best D-D offline Action L1 by checkpoint:

| Model | Best step | D-D Action L1 |
| --- | ---: | ---: |
| ACT-A | 25000 | 0.2589 |
| ACT-ABC | 125000 | 0.2537 |

The final ACT-ABC checkpoint reduces D-D Action L1 by about 0.0233 absolute, or 7.79% relative to ACT-A.

## Model Weights

Weights are not included in this repository because each run contains large checkpoint files.

Model weight download link:

```text
TODO: fill cloud storage URL
```

Expected final checkpoint paths on the training machine:

```text
/data/yzj/calvin_lerobot_act_hw2/checkpoints/act_calvin_a_clean_20260527_a_clean_bs64_200k_070441/checkpoints/200000
/data/yzj/calvin_lerobot_act_hw2/checkpoints/act_calvin_abc_clean_20260526_abc_clean_bs64_200k_022517/checkpoints/200000
```
