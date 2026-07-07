#!/bin/sh

# 在用seq去pretrain的时候，直接train的是agent1的条件分布，还需要用ind去train得到agent0的分布
datatypes=("random")  # "expert" "medium" "medium-replay" "random"
dataset_seed=0    # 0 1 2 3 4
mix=False

## algo selection
# difftype="SRPO"     # DQL
marltype="Seq"      # default Seq, Can train JAL, IND, CTDE
TASK="HalfCheetah-v2"
device=3

agent_idx=1 # 用于指定train 0 还是 1

# if train on mixed data: --mixed_data

for types in "${datatypes[@]}"
do
    echo "data types: $types, data number: $dataset_seed, mix datasets: $mix, training agents id: $agent_idx"
    python pretrain_behavior.py --data_type $types --dataset_num $dataset_seed --device $device --srpo_mode $marltype --seq_agent_id $agent_idx &
done