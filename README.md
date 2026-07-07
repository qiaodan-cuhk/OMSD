# OMSD: Offline MARL with Sequential Score Decomposition

This is the official implementation for the paper **"Offline MARL with Sequential Score Decomposition"**.

---

## 📢 Requirements & Installation

### 1. Environment Setup
- **Python**: 3.9
- **PyTorch**: `> 1.4` (Recommended: `pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3`)
- **Gym**: `==0.10.8`

### 2. MuJoCo Installation
The datasets are sampled from **MuJoCo 2.0**. Please note that using MuJoCo >= 2.1 may cause performance drops due to environment discrepancies.

We provide MuJoCo 2.00 files in `diffmarl/downloads/mujoco200` and the corresponding license `mjkey.txt`. Alternatively, you can download them via:
```bash
wget [https://www.roboti.us/download/mujoco200_linux.zip](https://www.roboti.us/download/mujoco200_linux.zip)
unzip mujoco200_linux.zip
```
Move files to your root directory and set the environment variables in your ~/.bashrc:

```bash
export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export MUJOCO_KEY_PATH=~/.mujoco${MUJOCO_KEY_PATH}
```

### 3. Multi-agent Environments
Install Multi-agent Particle Environments (MPE):

```bash
cd diffmarl/envs/multiagent-particle-envs
pip install -e .
```

For Multi-agent MuJoCo, please refer to the multiagent_mujoco directory provided in this repository.

## 📦 Datasets

The offline datasets for different tasks can be downloaded from the open-sourced links. Please see details in paper/Appendix:
Please decompress the downloaded datasets into the datasets/ folder.

## 🚀 Usage

To train the independent DiffusionMARL baseline, run:

```bash
python main.py --env_id <ENVIRONMENT_NAME> --data_type <DATA_TYPE> --dataset_num <DATASET_NUM>
```
Parameters:
--env_id: simple_spread / simple_tag / simple_world / HalfCheetah-v2

--data_type: random / medium-replay / medium / expert

--dataset_num: 0 / 1 / 2 / 3 / 4

Example:

```bash
python main.py --env_id HalfCheetah-v2 --data_type expert --dataset_num 0 --device 0 --seed 1
```

## 📚 Acknowledgement

Our codebase is built upon the open-source frameworks Diffusion-QL and OMAR. We thank the authors for their wonderful work.
