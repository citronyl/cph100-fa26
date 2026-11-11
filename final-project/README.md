# Can EEG decode moment-to-moment attention in children?

CPH 100 final project, UC Berkeley, Fall 2026. The assignment requirements are in [ASSIGNMENT.md](ASSIGNMENT.md).

## Question
Can EEG decode a child's trial-by-trial response time in the contrast change detection task, or does it only identify which child is being recorded?

## Data
Healthy Brain Network EEG (HBN-EEG), releases 1 to 11, 100 Hz BDF version from the EEG 2025 Challenge. Not stored in this repo.

## Compute environment
Runs were done on a Google Cloud Compute Engine VM:

| | |
|---|---|
| Machine | g2-standard-8 (8 vCPU, 32 GB RAM) |
| GPU | 1x NVIDIA L4 (23 GB), driver 580.178.04, CUDA 13.0 |
| Zone | us-west1-c |
| Provisioning | Standard (not Spot) |
| Image | Google Deep Learning `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` (Ubuntu 22.04) |
| PyTorch | 2.9.1+cu129 |
| Boot disk | 100 GB Hyperdisk Balanced |
| Data disk | 300 GB SSD persistent disk, mounted at `/mnt/data` (ext4) |

- Data lives in `/mnt/data/raw/R{1..11}_L100_bdf` (contrast change detection task only plus participant tables). Preprocessed windows go in `/mnt/data/processed`, checkpoints in `/mnt/data/checkpoints`, results in `/mnt/data/results`.
- Environment: Python venv `~/envs/cph100`, created with `--system-site-packages` so it reuses the image's PyTorch. Check it with `python check_requirements.py`.
- Long runs use `tmux`, which does not survive a VM stop, so scripts save checkpoints and write logs to disk.

## Setup

```bash
python3 -m venv ~/envs/cph100 --system-site-packages
source ~/envs/cph100/bin/activate
pip install -r requirements.txt
python check_requirements.py
```

PyTorch is not in `requirements.txt`. It comes from the GPU image (2.9.1+cu129).

### Getting the data
```bash
aws s3 sync s3://nmdatasets/NeurIPS25/R1_L100_bdf /mnt/data/raw/R1_L100_bdf --no-sign-request \
    --exclude "*" --include "*contrastChangeDetection*" --include "participants.*" \
    --include "dataset_description.json" --include "README" --include "code/*"
```
Repeat for R2 to R11. The data is the 100 Hz, 0.5 to 50 Hz band-passed version from the EEG 2025 Challenge.

## Layout
(fill in as the code appears)

## Results
(fill in)