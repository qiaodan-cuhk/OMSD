#!/bin/sh
datatypes=("medium-expert" "medium")  # "expert" "medium" "medium-replay" "medium-expert"

# optimal param
# expert: tau=0.7, temp=3,5,7
# medium-replay: tau=0.7/0.9, temp=3/7
# medium-expert: tau=0.5, temp=5/7 
# medium: tau=0.5, temp=0.5
# python /data/qiaodan/code/diffmarl/pretrain_critic.py --data_type "expert" --device 5 --tau 0.7 --temp 7
# python /data/qiaodan/code/diffmarl/pretrain_critic.py --data_type "medium-replay" --device 5 --tau 0.7 --temp 7
# python /data/qiaodan/code/diffmarl/pretrain_critic.py --data_type "medium-expert" --device 5 --tau 0.5 --temp 7
# python /data/qiaodan/code/diffmarl/pretrain_critic.py --data_type "medium" --device 5 --tau 0.7 --temp 7


## algo selection
# difftype="SRPO"     # DQL
marltype="JAL"      # Can train JAL, IND, CTDE    seq使用ctde/jal critic
TASK="Ant-v2"
devices=(1)  # 可用的GPU设备列表

# 添加tau和temp的搜索范围
tau_values=(0.7)
temp_values=(3.0) 

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
            
            echo "data types: $types, device: $current_device, tau: $tau, temp: $temp"
            python /data/qiaodan/code/diffmarl/pretrain_critic.py --env_id $TASK --data_type $types --device $current_device --srpo_mode $marltype --tau $tau --temp $temp &
            
            # 增加计数器
            counter=$((counter + 1))
        done
    done
done

# 等待所有后台进程完成
wait


