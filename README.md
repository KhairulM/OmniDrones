# RLInterceptorDrone

[![IsaacSim](https://img.shields.io/badge/Isaac%20Sim-5.1.0-green.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A fork of [OmniDrones](https://github.com/btx0424/OmniDrones) focused on training a multi-rotor **interceptor** drone via reinforcement learning. The repository ships an `Intercept` task in which a *pursuer* drone learns to catch a moving *evader* that follows scripted trajectories (hover, linear, circular).

> Heavy lifting (simulation, robot models, RL stack) comes from upstream OmniDrones. This README only covers what is needed to install the environment and train/play the `Intercept` task.

## Prerequisites

- Linux (tested on Ubuntu) with an NVIDIA RTX 5090
- NVIDIA driver compatible with **Isaac Sim 5.1.0**
- **Python 3.11** (Isaac Sim 5.1 ships and is tested with 3.11)
- A working installation of [NVIDIA Isaac Sim 5.1.0](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/index.html)

## Installation

The project is laid out as a standard Python package at the repo root. Install it into a Python 3.11 virtual environment that has access to Isaac Sim's Python modules.

### 1. Clone

```bash
git clone git@github.com:KhairulM/RL-Interceptor-Drone-v2.git RLInterceptorDrone
cd RLInterceptorDrone
```

### 2. Create the virtual environment

```bash
python3.11 -m venv --prompt="interceptor" .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install Isaac Sim Python packages

Follow the [official Isaac Sim 5.1 pip install instructions](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html). In short, with the venv active:

```bash
pip install isaacsim==5.1.* --extra-index-url https://pypi.nvidia.com
```

### 4. Install this package (editable)

From the repo root:

```bash
pip install -e .
```

This installs `omni_drones` and its Python dependencies (`hydra-core`, `omegaconf`, `torchrl`, `tensordict`, `wandb`, `numpy`, `scipy`, `imageio`, `moviepy`, ...).

### 5. Verify

```bash
python -c "import omni_drones; print(omni_drones.__file__)"
```

It should print a path inside this repository.

## Running the Intercept task

The `Intercept` task lives at [omni_drones/envs/single/intercept.py](omni_drones/envs/single/intercept.py) and is configured via [cfg/task/Intercept.yaml](cfg/task/Intercept.yaml). A pursuer drone (`RateController`) learns to catch an evader (`LeePositionController`) that follows scripted hover/linear/circular trajectories.

All commands below assume the venv is activated and you are at the repo root.

### Train (headless, recommended)

```bash
cd scripts
python train.py task=Intercept headless=true env.num_envs=256
```

Useful overrides (Hydra syntax — `key=value`, dotted paths allowed):

| Override | Purpose |
|---|---|
| `task=Intercept` | Select the Intercept task config |
| `algo=mappo` | RL algorithm (default; see [cfg/algo/](cfg/algo/)) |
| `env.num_envs=256` | Parallel environments (start small; see note below) |
| `task.action_transform=rate` | Use the body-rate action transform (default) |
| `task.action_transform=null` | Disable the rate transform (raw rotor commands) |
| `headless=true` | Run without the Isaac Sim GUI (faster) |
| `total_frames=10_000_000` | Cap training frames |
| `wandb.mode=disabled` | Disable Weights & Biases logging |
| `seed=0` | RNG seed |

Outputs (checkpoints, configs, videos) are written under `scripts/outputs/<date>/<time>/` by Hydra.

### Play / evaluate a trained policy

```bash
python scripts/play.py task=Intercept headless=false \
    checkpoint_path=scripts/outputs/<date>/<time>/checkpoint_final.pt
```

Set `headless=false` to watch the rollout in the Isaac Sim viewport.

### Tuning the Intercept scenario

Common knobs in [cfg/task/Intercept.yaml](cfg/task/Intercept.yaml):

- `pursuer.model` / `pursuer.controller` — drone model and low-level controller of the interceptor
- `evader.model` / `evader.controller` — drone model and controller of the target
- `evader.trajectories` — list of trajectories sampled per env (`hover`, `linear`, `circular`)
- `evader.speed_range`, `evader.spawn_distance_range`, `evader.bounds`, `evader.boundary_mode` — evader motion limits
- `success_radius`, `reset_thres`, `reward_distance_scale` — reward / termination shaping
- `env.num_envs`, `env.max_episode_length`, `env.env_spacing` — sim batching

### Notes

- On RTX 5090 / Isaac Sim 5.1, the verified-stable Intercept setup uses a **homogeneous Hummingbird pursuer + Hummingbird evader** scene (already the default). Mixing different drone models in the same scene at high `env.num_envs` has been observed to crash the PhysX GPU backend.
- Start with `env.num_envs=256` and scale up only after confirming stability on your hardware.

## Repository layout

```
cfg/                Hydra configs (task/, algo/, base/)
omni_drones/        Python package: envs, robots, controllers, learning, sensors, utils
scripts/            Entry points: train.py, play.py (Hydra apps reading cfg/)
examples/           Standalone demo scripts
docs/               Sphinx documentation source
```

## Upstream

This repo descends from [btx0424/OmniDrones](https://github.com/btx0424/OmniDrones) (configured as the `upstream` remote). For the broader feature set, multi-agent benchmarks, and original documentation, see the [upstream docs](https://omnidrones.readthedocs.io/en/latest/).

## Citation

If you use this work, please cite the original OmniDrones paper:

```bibtex
@misc{xu2023omnidrones,
    title={OmniDrones: An Efficient and Flexible Platform for Reinforcement Learning in Drone Control},
    author={Botian Xu and Feng Gao and Chao Yu and Ruize Zhang and Yi Wu and Yu Wang},
    year={2023},
    eprint={2309.12825},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```

## Acknowledgement

Built on top of [OmniDrones](https://github.com/btx0424/OmniDrones) and [NVIDIA Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html). Some abstractions are inspired by [Isaac Lab](https://github.com/isaac-sim/IsaacLab).
