# Training notes

Run behavior pretraining (`pretrain_behavior.py`), IQL pretraining
(`pretrain_critic.py`), and policy training (`main.py`) as independent stages.
Use the same algorithm, dataset and agent configuration across stages.

## Data and parallel jobs

For MPE, choose one data-generation seed export, such as `seed_0_data`, and
keep it fixed for the experiment. The training `--seed` is independent of the
dataset seed selected by `--dataset-num`. Tag/world learn the three predators;
evaluation requires a matching prey checkpoint.

MADiff/OG-MARL uses packed episode NumPy exports. Use the MuJoCo 2.1 environment
and matching agent partition: `2halfcheetah` = `2x3`, `2ant` = `2x4`, and
`4ant` = `4x2`. The adapter removes the one-hot agent-ID prefix; use
`--observation-prefix 0` for exports without that prefix.

`--pretrain-agent-id` trains one physical dataset agent while retaining all
learner data needed for sequential conditions. Use separate output directories
for concurrent jobs. Supply complete per-agent checkpoint lists in dataset-agent
order. OMSD/CTDE use one joint critic; IND critics may also train by agent.
The [Bash examples](../bash/README.md) show these workflows.

## Checkpoint selection

Select behavior and IQL checkpoints by the resulting policy performance.
It is useful to try several intermediate pretraining checkpoints rather than
always using the final files. In some experiments, checkpoints around epochs
160–180 of a 200-epoch pretraining run give better policy performance than the
checkpoint at epoch 200, for either diffusion or IQL.

Current `*_stepN.pt` filenames count optimizer updates, not epochs. Choose the
files corresponding to the desired point in pretraining and pass their paths
explicitly with `--behavior-checkpoint` and `--critic-checkpoint`. Existing
matching weights and complete per-agent file lists can also be loaded directly.
Keep network architecture, agent order and conditioning settings consistent.

## Training parameters

| Stage | Default pretraining budget | Batch size |
| --- | --- | --- |
| Diffusion behavior | 200 epochs × 10,000 updates | 512 |
| IQL critic | 200 epochs × 5,000 updates | 1024 |
| Policy | Set with `--num-steps` | 512 |

An epoch is a fixed number of random replay-buffer updates. Adjust `--tau`
for the IQL expectile and `--beta` for policy regularization according to the
task and dataset quality. The MPE spread expert example uses tau 0.5 and beta
0.001. The OG-MARL example provides a starting configuration; consult the
paper and tune the selected experiment.

Use `--help` on each training file for stage-specific options, or use a flat
YAML configuration through `--config`. Evaluation uses the matching simulator;
set `--eval-interval` to a positive value to enable it.
