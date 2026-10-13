# Independent experiment stages

Each script starts one training stage. Diffusion and IQL pretraining are
independent; policy training starts after the selected checkpoints are ready.
The examples use existing local data and download nothing. Activate the Python
environment first. Relative paths are resolved from the repository root.

| Example | Settings | Diffusion | IQL | Policy |
| --- | --- | --- | --- | --- |
| OMAR `simple_spread`, expert | `mpe/settings.sh` | `mpe/pretrain_diffusion.sh` | `mpe/pretrain_iql.sh` | `mpe/train.sh` |
| MADiff/OG-MARL `2halfcheetah`, good | `mamujoco/settings.sh` | `mamujoco/pretrain_diffusion.sh` | `mamujoco/pretrain_iql.sh` | `mamujoco/train.sh` |

Settings are shared within each example. MPE expert uses critic expectile 0.5,
policy beta 0.001 and 100,000 policy updates. The OG-MARL example fixes the `2x3`
partition and uses illustrative values 0.7 / 0.01 with 500,000 policy updates;
adjust them for the selected experiment. Pretraining uses the
default stage budgets and batches. See [reproduction notes](../docs/reproduction.md).

## Choose the MPE dataset once

OMAR exports contain separate data-generation seeds, such as `seed_0_data`.
Choose one available seed for an experiment, randomly if desired, and keep that
same directory for every diffusion agent, IQL and policy stage. The loader does
not randomly switch seed folders. `OMSD_DATASET_NUM` / `--dataset-num` selects
the folder; `OMSD_SEED` / `--seed` controls model initialization and training
sampling. These two seeds have different meanings.

With data in the default layout:

```bash
export OMSD_DATASET_NUM=0
export OMSD_SEED=42
bash bash/mpe/pretrain_diffusion.sh
bash bash/mpe/pretrain_iql.sh
bash bash/mpe/train.sh
```

The first two commands may run in separate terminals or on separate GPUs. Their
order is unrestricted. Set `OMSD_DATA_PATH` for a custom location; it takes
precedence over the default `datasets/simple_spread/expert/seed_0_data` path.
Keep `OMSD_DATASET_NUM` consistent with that path as the recorded dataset label.

The MuJoCo example uses an explicit OG-MARL episode export:

```bash
export OMSD_DATA_PATH=/path/to/2halfcheetah/good
bash bash/mamujoco/pretrain_diffusion.sh
bash bash/mamujoco/pretrain_iql.sh
bash bash/mamujoco/train.sh
```

## Parallel diffusion jobs

By default, one diffusion job trains all selected learners. Set
`OMSD_PRETRAIN_AGENT_ID` to train just that physical dataset agent. The job still
loads the full aligned learner data: OMSD needs the preceding agents' dataset
actions for its sequential condition. This is different from `--agent-ids`,
which chooses which agents belong to the learning problem.

For the three spread agents, on three available GPUs:

```bash
pids=()
for agent_id in 0 1 2; do
  OMSD_PRETRAIN_AGENT_ID="$agent_id" OMSD_DEVICE="$agent_id" \
    bash bash/mpe/pretrain_diffusion.sh &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then exit 1; fi
```

Each job writes to its own `behavior/agent<ID>/` directory. After all jobs
succeed, supply their chosen checkpoints in dataset-agent order:

```bash
experiment=outputs/mpe/simple_spread/expert/OMSD/dataset0/seed42
bash bash/mpe/train.sh \
  --behavior-checkpoint "$experiment/behavior/agent0/behavior.pt" \
    "$experiment/behavior/agent1/behavior.pt" \
    "$experiment/behavior/agent2/behavior.pt" \
  --critic-checkpoint "$experiment/critic/critic.pt"
```

SEQ behavior pretraining does not depend on another agent's behavior weights,
so IQL can also run while these jobs run, if GPU memory permits. OMSD/CTDE have
one joint IQL critic and must pretrain it as a whole. IND has independent
critics and also supports `OMSD_PRETRAIN_AGENT_ID` in `pretrain_iql.sh`; provide
the full critic file list in dataset-agent order when training policy.

## Settings, outputs and checkpoint selection

Edit each example's `settings.sh`, or set environment overrides:

| Variable | Meaning |
| --- | --- |
| `OMSD_PYTHON` | Python executable from the installed environment |
| `OMSD_ALGORITHM` | `OMSD` (default), `CTDE`, or `IND`, used by every stage |
| `OMSD_DEVICE` | GPU 0 by default; select another GPU index |
| `OMSD_SEED` | Model/sampling RNG, default 42 |
| `OMSD_DATA_PATH` | Existing dataset directory |
| `OMSD_DATA_TYPE`, `OMSD_DATASET_NUM` | Dataset quality and MPE data-generation seed |
| `OMSD_TAU`, `OMSD_BETA` | Critic expectile and policy regularization |
| `OMSD_NUM_STEPS` | Policy update budget |
| `OMSD_PRETRAIN_AGENT_ID` | Optional single physical agent for pretraining |
| `OMSD_EVAL_INTERVAL` | 0 disables evaluation; a positive value enables it |
| `OMSD_OUTPUT_DIR` | Shared experiment root for all stages |
| `OMSD_BEHAVIOR_CHECKPOINT`, `OMSD_CRITIC_CHECKPOINT` | Explicit policy inputs, overriding the default final component files |

The default experiment root includes profile, task, quality, algorithm, MPE
seed folder and training seed. Stages write to `behavior/`, `critic/`, and
`train/`; single-agent pretraining adds `agent<ID>/`. Existing stage outputs
are not overwritten. Choose a new `OMSD_OUTPUT_DIR` for a repeated experiment.
The final checkpoint paths are defaults, not a mandatory epoch selection.

To use intermediate or existing weights, pass explicit files to `train.sh`,
or edit its checkpoint arrays in `settings.sh`. No pretraining starts when
running `train.sh`. Complete new per-agent containers and matching legacy bare
weights are accepted, without model conversion. Keep architecture, dataset and
conditional order matched. Extra stage-specific CLI arguments pass through to
the Python file, for example `pretrain_iql.sh --tau 0.7`.

Evaluation is disabled by default. Set `OMSD_EVAL_INTERVAL=10000` to log returns
during policy training. MPE uses the included environment. The MADiff/OG-MARL
MuJoCo examples require a separate environment with the main branch's dependencies
and matching MuJoCo 2.1 engine; see the
[main installation instructions](https://github.com/qiaodan-cuhk/OMSD/tree/main#requirements-and-installation).
