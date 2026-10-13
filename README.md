# OMSD: Offline Multi-agent Reinforcement Learning via Sequential Score Decomposition

Official implementation of [the paper](https://arxiv.org/abs/2505.05968).

## Requirements and installation

Use Linux and Python 3.9. The recommended GPU environment is PyTorch 1.10.1
with CUDA 11.3:

```bash
conda create -n omsd-omiga python=3.9 -y
conda activate omsd-omiga
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
python -m pip install -r requirements.txt
```

For newer GPUs, install a compatible CUDA-enabled PyTorch build before the
requirements. The requirements support PyTorch 1.10.1 and later 1.x/2.x versions.
Training defaults to GPU 0; select another GPU with `--device 1` or `--device-id 1`.

MPE is included in this repository and needs no separate installation. This
branch uses Gym 0.17.2 and `mujoco-py==2.0.2.8` for the OMIGA environments.

For OMIGA evaluation, install the MuJoCo 2.0 engine under
`~/.mujoco/mujoco200` following the
[mujoco-py guide](https://github.com/openai/mujoco-py#install-mujoco), then run:

```bash
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco200/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

See [OMIGA installation and training](docs/omiga.md) for task settings. To evaluate
MADiff/OG-MARL MuJoCo 2.1, use the `main` branch's `requirements.txt` in a separate
environment with the MuJoCo 2.1 engine. Offline training supports both dataset
formats; simulator versions must match the dataset for evaluation.

## Algorithms

Select the same `--algorithm` in all three stages; `OMSD` is the default.

| Algorithm | Behavior model | IQL critic |
| --- | --- | --- |
| `OMSD` | Sequential conditional diffusion | Joint-action centralized |
| `CTDE` | Independent diffusion | Joint-action centralized |
| `IND` | Independent diffusion | Per-agent independent |

Behavior models condition on local observations; OMSD also uses preceding
agents' actions. Execution policies use only their own observations.

## Datasets

Download datasets following the [OMAR](https://github.com/ling-pan/OMAR) and
[MADiff](https://github.com/zbzhu99/madiff) instructions.

- **MPE:** `simple_spread`, `simple_tag`, and `simple_world`, using per-agent NumPy exports.
- **Multi-agent MuJoCo 2.1:** `2halfcheetah`, `2ant`, and `4ant`, using MADiff/OG-MARL episode exports.
- **OMIGA:** `HalfCheetah-v2`, `Ant-v2`, and `Hopper-v2`, using HDF5 exports;
  see [OMIGA datasets and ablations](docs/omiga.md).

For MPE, choose one available dataset seed for an experiment, randomly if
desired, such as `seed_0_data`. Use that same directory for all behavior agents,
IQL and policy training. `--dataset-num` selects the data-generation seed folder
in the default layout; `--seed` controls training randomness.

## Training

The three stages have independent entry points: `pretrain_behavior.py`,
`pretrain_critic.py`, and `main.py`. Behavior and IQL pretraining can run in
parallel; policy training uses the selected pretrained checkpoints.

The Bash examples share settings within each environment and start one stage
per command. For MPE `simple_spread`, expert:

```bash
export OMSD_DATA_PATH=/path/to/simple_spread/expert/seed_0_data
export OMSD_DATASET_NUM=0
bash bash/mpe/pretrain_diffusion.sh
bash bash/mpe/pretrain_iql.sh
bash bash/mpe/train.sh
```

For MADiff/OG-MARL `2halfcheetah`, good (use the main branch environment for evaluation):

```bash
export OMSD_DATA_PATH=/path/to/2halfcheetah/good
bash bash/mamujoco/pretrain_diffusion.sh
bash bash/mamujoco/pretrain_iql.sh
bash bash/mamujoco/train.sh
```

Edit `bash/mpe/settings.sh` or `bash/mamujoco/settings.sh`, or use environment
overrides such as `OMSD_ALGORITHM=CTDE`, `OMSD_DEVICE=1`, and `OMSD_SEED=42`.
Diffusion can also be pretrained by agent:

```bash
OMSD_PRETRAIN_AGENT_ID=1 OMSD_DEVICE=1 bash bash/mpe/pretrain_diffusion.sh
```

Each agent job retains the full sequential conditioning data and writes its
own checkpoint. Supply all agent files in dataset-agent order when training
policy. OMSD/CTDE use one joint IQL critic; IND critics can also be trained
independently by agent. See [Bash examples](bash/README.md) for parallel jobs and
checkpoint lists.

## Outputs and checkpoint selection

Stages save checkpoints, configuration and metrics under the experiment output
directory. Set `OMSD_OUTPUT_DIR` in Bash or `--output-dir` in Python to choose it.
Use a new directory for a repeated run.

Policy training accepts explicit `--behavior-checkpoint` and
`--critic-checkpoint` paths. Try intermediate pretraining checkpoints when
tuning performance; the final checkpoint is not always the best choice.
See [training notes](docs/reproduction.md) for checkpoint selection and parameters.

To evaluate during policy training, set `OMSD_EVAL_INTERVAL=10000` in Bash or
pass `--eval-interval 10000` to `main.py`. Tag/world evaluation also requires
`--opponent-checkpoint` for the prey policy.

## OMIGA and ablations

[`main`](https://github.com/qiaodan-cuhk/OMSD/tree/main) provides the MPE and
MADiff/OG-MARL workflow. The
[`omiga` branch](https://github.com/qiaodan-cuhk/OMSD/tree/omiga) adds OMIGA datasets
and environments, GMM behavior models, and fixed-order ablations using the same
training core.

## Acknowledgements

We thank [OMAR](https://github.com/ling-pan/OMAR),
[MADiff](https://github.com/zbzhu99/madiff),
[Diffusion-QL](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL),
and the multi-agent environment projects for their open-source implementations.

## OMIGA runtime on this branch

See [OMIGA and ablation instructions](docs/omiga.md) for the separate simulator
environment, dataset configuration, GMM pretraining and fixed-order experiments.

## Citation

If you find this code or our work useful in your research, please cite:

```bibtex
@misc{qiao2026offlinemultiagentreinforcementlearning,
      title={Offline Multi-agent Reinforcement Learning via Sequential Score Decomposition},
      author={Dan Qiao and Wenhao Li and Shanchao Yang and Hongyuan Zha and Baoxiang Wang},
      year={2026},
      eprint={2505.05968},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.05968},
}
```
