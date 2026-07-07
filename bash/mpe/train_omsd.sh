#!/bin/sh
# simple spread 数据集用 0 0 0 2
# simple tag 4 0 4 0
# simple world 1 0 1 2

TASK="simple_spread"
declare -A dataset_seeds=(
    ["expert"]=0
    ["medium"]=0
    ["medium-replay"]=0
    ["random"]=2
)

seeds=(42)  # 改为数组形式
# seeds=(42 99 1000)

device=(1 2 3 4 5)
# beta=(0.01 0.05 0.1 0.2 0.3 0.5 0.7)   # 0.001 0.002 0.005 0.007 0.01 0.05 0.1 0.2 0.3 0.5 0.7
# beta=(0.01)


# 为每种数据类型定义对应的beta值范围
declare -A type_betas=(
    ["expert"]="0.001 0.003 0.005 0.01 0.05"
    ["medium"]="0.01 0.02 0.03 0.04 0.05"
    ["random"]="0.3 0.4 0.5 0.6 0.7"
)
datatypes=("expert" "medium" "random")  # "expert" "medium-replay" "medium" "random"

diffusion_epoch_num=(199)           # 149
critic_epoch_num=(499)    # 99

difftype="SRPO"
marltype="SEQ"
counter=0
trainsteps=100000
annealing_epochs=10

# beta=(0.001 0.005 0.01 0.02 0.05 0.1 0.2 0.5)  # medium-replay 在0.02附近表现较好
# beta=(0.01 0.015 0.02 0.025 0.03 0.04)
# 0.05 - 0.25附近 for random
# beta=(0.02)
# beta=(0.0001 0.001 0.01 0.02 0.05 0.1 0.2 0.5 1)

for types in "${datatypes[@]}"
do
    # 获取当前数据类型对应的beta值
    IFS=' ' read -r -a current_betas <<< "${type_betas[$types]}"
    
    for i in "${seeds[@]}"
    do
        for beta_i in "${current_betas[@]}"
        do
            for critic_i in "${critic_epoch_num[@]}"
            do
                for diff_i in "${diffusion_epoch_num[@]}"
                do
                    dataset_num=${dataset_seeds[$types]}
                    current_device=${device[$((counter % ${#device[@]}))]}
                    

                    echo "seed: $i, datatypes: $types, data number: $dataset_num"
                    echo "beta: $beta_i, device: $current_device, critic: $critic_i, diff: $diff_i"
                    
                    python main.py --env_id $TASK \
                                --data_type $types \
                                --dataset_num $dataset_num \
                                --seed $i \
                                --device $current_device \
                                --difftype $difftype \
                                --marltype $marltype \
                                --num_steps $trainsteps \
                                --beta $beta_i \
                                --n_policy_epochs $annealing_epochs \
                                --diff_epoch $diff_i \
                                --critic_epoch $critic_i &

                    counter=$((counter + 1))  # 更新counter

                done
            done
        done
    done
done

wait
