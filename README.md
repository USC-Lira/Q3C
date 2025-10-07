# Actor-Free Continuous Control via Structurally Maximizable Q-Functions

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
<!-- [![Paper](https://img.shields.io/badge/arXiv-2501.01234-b31b1b.svg)](https://arxiv.org/abs/2501.01234) -->

This repository provides the **official implementation** of the paper:

> **Actor-Free Continuous Control via Structurally Maximizable Q-Functions**  
> Yigit Korkmaz*, Urvi Bhuwania*, Ayush Jain†, Erdem Bıyık†, NeurIPS 2025 (to appear)*

We introduce **Q3C**, a framework for continuous control without an explicit actor network by leveraging structurally maximizable Q-functions for stable and efficient policy learning.

## Installation

### Create a new Conda environment
```bash
conda create --name q3c_env python=3.10
```

### Activate the environment
```bash
conda activate q3c_env
```

### Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

**Note:**  MuJoCo must be installed separately, as it is required for all benchmark environments.

This repository uses **[Hydra](https://hydra.cc/)** for hierarchical configuration management.  
The file `configs/hydra_sb3_q3c_hyperparams.yaml` defines environment-specific hyperparameters.


## Running Experiments

To train and log results to **Weights & Biases**, run:

```bash
python scripts/run_experiment.py wandb.entity=YOUR_WANDB_USERNAME train.environment=ENV_NAME
```

- Replace `YOUR_WANDB_USERNAME` with your W&B account name.  
- Replace `ENV_NAME` with your target environment (e.g., `Hopper-v4`, `HalfCheetah-v4`, `Walker2d-v4`).  

All metrics, losses, and rollout statistics will be automatically logged to your W&B project.


## Citation

If you find this repository useful for your research, please cite:

```bibtex
@inproceedings{korkmaz2025actor,
  title={Actor-Free Continuous Control via Structurally Maximizable Q-Functions},
  author={Korkmaz, Yigit and Bhuwania, Urvi and Jain, Ayush and Bıyık, Erdem},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

