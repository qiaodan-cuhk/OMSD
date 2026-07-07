# 在用seq去pretrain的时候，直接train的是agent1的条件分布，还需要用ind去train得到agent0的分布
# simple spread 数据集用 0 0 0 2
# simple tag 4 0 4 0
# simple world 1 0 1 2

#!/bin/bash

TASK="simple_tag"
marltype="IND"

# 可用GPU
devices=(1 2 3 4 5)

# 不同数据类型对应的数据seed
declare -A dataset_seeds=(
    ["expert"]=4
    ["medium"]=0
    ["random"]=0
)

datatypes=("expert" "medium" "random")

# 关键：要并行训练的agent id列表
# simple_tag 如果是4个adversary，就写 0 1 2 3
agent_ids=(0 1 2)

counter=0
for types in "${datatypes[@]}"; do
    seed=${dataset_seeds[$types]}

    for aid in "${agent_ids[@]}"; do
        device=${devices[$((counter % ${#devices[@]}))]}
        echo "data type: $types, data seed: $seed, agent_id: $aid, device: $device, mode: $marltype"

        python /data/qiaodan/code/diffmarl/pretrain_behavior.py \
            --env_id "$TASK" \
            --data_type "$types" \
            --dataset_num "$seed" \
            --device "$device" \
            --srpo_mode "$marltype" \
            --seq_agent_id "$aid" &

        counter=$((counter + 1))
    done
done

wait