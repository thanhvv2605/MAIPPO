# MAIPPO: Multi-Agent IPPO for Charging Scheduling in WRSNs

Research code for coordinating several **mobile chargers** in a **wireless rechargeable sensor
network (WRSN)** with multi-agent deep reinforcement learning. Each charger learns where to go
and how long to charge so that the network stays alive as long as possible.

> A paper based on this work is in preparation. This repository is a summary; full method
> details and results will be linked after publication.

## Problem

Sensor nodes drain their batteries at different rates. A small fleet of mobile chargers must
decide, continuously and without a central planner, **which location to move to** and **how long
to charge there**. Bad coordination means chargers cluster in the same area while nodes elsewhere
die, which can break network coverage.

## Approach

```
              ┌──────────────── SimPy WRSN simulator ────────────────┐
              │ sensor nodes · base station · targets · energy model │
              └──────────────────────────┬───────────────────────────┘
                                         │ per-agent observation
                                         ▼
     4-channel heatmap (H × W):  node energy demand · own battery
                                 · teammates charging · teammates moving
                                         │
                       ┌─────────────────┴─────────────────┐
                       ▼                                   ▼
              U-Net actor (per agent)              CNN critic
                       │
                       ▼
     continuous action (x, y, charging time)  ──►  mobile charger executes
```

- **Environment**: event-driven simulator (SimPy) wrapped as a `gym.Env`, with scenarios from
  50 to 200 nodes (`physical_env/network/network_scenarios/`).
- **Observation**: each agent sees the network as Gaussian-smoothed spatial maps rather than a
  node list, so the same policy works for any network size.
- **Action**: a continuous target location plus a charging duration, normalised to `[0, 1]`.
- **Reward**: priority-weighted energy delivered to nodes close to depletion, which pushes agents
  to cover different areas instead of competing for the same nodes.
- **Learning**: Independent PPO (shared hyper-parameters in `alg_args/ippo.yaml`): GAE,
  advantage normalisation, clipped value loss, learning-rate and entropy decay, KL early stop.

## Repository layout

```
rl_env/WRSN.py          gym environment: observations, reward, multi-agent stepping
physical_env/           network, sensor node, base station and mobile-charger models + scenarios
controller/ippo/        IPPO algorithm, U-Net actor, CNN critic
runner/IPPO.py          training entry point
runner/check*.py        evaluation of trained policies
draw/                   training-curve plotting
```

## Run

```bash
conda create --name wrsn python=3.10
conda activate wrsn
pip install -r requirements.txt
python runner/IPPO.py
```

A CUDA GPU is recommended for training.
