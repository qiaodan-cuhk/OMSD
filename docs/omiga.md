# OMIGA datasets and ablations

This branch adds OMIGA data and environments, GMM behavior models and fixed-order
ablations to the shared OMSD training workflow.

## Installation

Use a separate environment for OMIGA:

```bash
conda create -n omsd-omiga python=3.9 -y
conda activate omsd-omiga
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
python -m pip install -r requirements.txt
```

Install a matching CUDA-enabled PyTorch build for newer GPUs. The OMIGA runtime
uses Gym 0.17.2 and `mujoco-py==2.0.2.8`, with the external MuJoCo 2.0 engine and
system libraries. Keep it separate from the main workflow's MuJoCo 2.1 runtime.
Place the MuJoCo 2.0 `mjkey.txt` at `~/.mujoco/mjkey.txt`, or set
`MUJOCO_PY_MJKEY_PATH` to its location.
The included environment source is from the official
[OMIGA project](https://github.com/ZhengYinan-AIR/OMIGA); its
[source notice](../envs/ma_mujoco/NOTICE.md) records the version and attribution.

## OMIGA training

| Task | Agent partition |
| --- | --- |
| `HalfCheetah-v2` | `6x1` |
| `Ant-v2` | `2x4` |
| `Hopper-v2` | `3x1` |

Use OMIGA's `s/o/a/r/d` HDF5 datasets with `configs/omiga.yaml`. For example:

```bash
OMSD_DATA_PATH=/path/to/HalfCheetah-v2-6x1-expert.hdf5

python pretrain_behavior.py --config configs/omiga.yaml \
  --data-path "$OMSD_DATA_PATH" --output-dir outputs/omiga-expert/behavior

python pretrain_critic.py --config configs/omiga.yaml \
  --data-path "$OMSD_DATA_PATH" --output-dir outputs/omiga-expert/critic

python main.py --config configs/omiga.yaml --data-path "$OMSD_DATA_PATH" \
  --behavior-checkpoint outputs/omiga-expert/behavior/behavior.pt \
  --critic-checkpoint outputs/omiga-expert/critic/critic.pt \
  --output-dir outputs/omiga-expert/train
```

Behavior and critic pretraining are independent. Keep dataset, agent partition,
algorithm and checkpoint architecture consistent. Behavior can also train by
agent with `--pretrain-agent-id`; pass the complete file list in dataset-agent
order to policy training. See [checkpoint selection](reproduction.md#checkpoint-selection)
for intermediate checkpoint tuning.

## GMM ablation

GMM is used only for ablation. Its pretraining and policy files are separate;
reuse the matching IQL critic from normal `pretrain_critic.py`.

```bash
python ablation/gmm/pretrain_behavior.py --algorithm OMSD --profile mpe \
  --env-id simple_tag --data-path /path/to/predator-prey/data \
  --n-gmm-components 8 --gmm-hidden-dim 512 \
  --output-dir outputs/gmm/behavior

python ablation/gmm/train.py --algorithm OMSD --profile mpe \
  --env-id simple_tag --data-path /path/to/predator-prey/data \
  --n-gmm-components 8 --gmm-hidden-dim 512 \
  --behavior-checkpoint outputs/gmm/behavior/behavior.pt \
  --critic-checkpoint /path/to/joint_critic.pt \
  --output-dir outputs/gmm/train
```

Match architecture, critic, training budgets and evaluation settings when
comparing GMM with diffusion. GMM policy regularization uses the negative
log-density score in the minimized loss and normalizes its magnitude by default.
Use `--no-gmm-score-normalize` for a raw-score experiment. Evaluation on tag/world
also requires a matching prey checkpoint.

## Fixed-order ablation

Use `ablation/order/pretrain_behavior.py` and `ablation/order/train.py` with the
same `--conditional-order` and matching behavior checkpoints. The joint IQL
critic can be reused. For example, Hopper supports `0-1-2`, `0-2-1` and `2-1-0`.
`--conditional-order random` selects one seeded fixed permutation for the run.
The standard workflow uses the natural agent order without an extra parameter.
