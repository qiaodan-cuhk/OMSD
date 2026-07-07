#!/bin/sh

# expert 用 datanum 3最好，medium+replay用datanum 1最好

datatypes=("random")  # "expert" "medium" "medium-replay" "random"
dataset_seed=0   # 1 2 3
mix=False

## algo selection
# difftype="SRPO"     # DQL
marltype="JAL"      # Can train JAL, IND, CTDE    seq使用ctde/jal critic
TASK="HalfCheetah-v2"
devices=(1 2 3 4 5)  # 可用的GPU设备列表

# 添加tau和temp的搜索范围
tau_values=(0.5 0.7 0.9)
temp_values=(0.5 1.0 3.0 5.0 7.0) 

# 初始化计数器
counter=0
total_devices=${#devices[@]}

# if train on mixed data: --mixed_data

for types in "${datatypes[@]}"
do
    for tau in "${tau_values[@]}"
    do
        for temp in "${temp_values[@]}"
        do
            # 计算当前应该使用的设备索引
            device_index=$((counter % total_devices))
            current_device=${devices[$device_index]}
            
            echo "data types: $types, device: $current_device, data number: $dataset_seed, mix datasets: $mix, tau: $tau, temp: $temp"
            python pretrain_critic.py --env_id $TASK --data_type $types --dataset_num $dataset_seed --device $current_device --srpo_mode $marltype --tau $tau --temp $temp &
            
            # 增加计数器
            counter=$((counter + 1))
        done
    done
done

# 等待所有后台进程完成
wait