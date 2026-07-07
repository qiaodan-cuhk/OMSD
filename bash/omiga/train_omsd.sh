#!/bin/sh
# seeds=(42 99 1000 9999 23477)   #  100
seeds=(42)

## algo seletion
difftype="SRPO"     # DQL
marltype="SEQ"      
TASK="Ant-v2"
trainsteps=1000000

devices=(0 1 2 3 4 5)  # 可用的GPU设备列表

# 目前只跑了medium和expert
# beta=(0.001 0.005)
# types=("expert") # "expert" "random" "medium-replay"


# beta=(0.001 0.03)
# types=("medium-expert")

beta=(0.001 0.003)
types=("medium-replay")
# # # 这个数据集有问题，先不跑了

# beta=(0.005 0.01)
# types=("medium")


diffusion_epoch_num=179
critic_epoch_num=179

pretrain_model_path="/data/qiaodan/code/diffmarl/pretrain/omiga/"


# conditional_order="2-1-0"   # 0-2-1
# conditional_order_list=("0-2-1" "2-1-0")

# 初始化计数器
counter=4
total_devices=${#devices[@]}

for i in "${seeds[@]}"
do
    for beta_i in "${beta[@]}"
    do
        
        # 计算当前应该使用的设备索引
        device_index=$((counter % total_devices))
        current_device=${devices[$device_index]}
        
        echo "seed: $i, datatypes: $types, beta: $beta_i, device: $current_device"

        CUDA_VISIBLE_DEVICES=$current_device \
        python /data/qiaodan/code/diffmarl/main.py --env_id $TASK --data_type $types --seed $i --device 0 --difftype $difftype --marltype $marltype --pretrain_model_path $pretrain_model_path --diff_epoch $diffusion_epoch_num --critic_epoch $critic_epoch_num --num_steps $trainsteps --beta $beta_i &
        
        # 增加计数器
        counter=$((counter + 1))
    done
done

# 等待所有后台任务完成
wait