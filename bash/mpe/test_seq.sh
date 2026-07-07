#!/bin/sh
TASK="simple_tag"
declare -A dataset_seeds=(
    ["expert"]=4
    ["medium"]=0
    ["medium-replay"]=4
    ["random"]=0
)

seeds=(42)  # 改为数组形式
device=(5)  # 使用所有可用的 GPU
beta=(0.01)
datatypes=("expert")  # "expert" "medium-replay" "medium" "random"

diffusion_epoch_num=(149)           # 149
critic_epoch_num=(119)    # 59 79 99｜ 119 139 159 179  

difftype="SRPO"
marltype="SEQ"
counter=0
trainsteps=100000
annealing_epochs=10

for types in "${datatypes[@]}"
do
    for i in "${seeds[@]}"
    do
        for beta_i in "${beta[@]}"
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

                    counter=$((counter + 1))  # 更新 counter
                done
            done
        done
    done
done

wait