#!/bin/sh
seeds=(42 100)
datatypes=("medium" "medium-replay")  # "expert" "random"
datanums=0    # 1 2 3

## algo seletion
difftype="SRPO"     # DQL
marltype="JAL"      # JAL, VD, SEQ
TASK="HalfCheetah-v2"
device=1


for i in "${seeds[@]}"
do
    for types in "${datatypes[@]}"
    do
        # actor/critic load path
        critic_load_path="/home/qiaodan/Code/diffmarl/SRPO_premodels/HalfCheetah-v2_${types}"
        diffusion_load_path="/home/qiaodan/Code/diffmarl/SRPO_premodels/HalfCheetah-v2_${types}"

        for num in "${datanums[@]}"
        do
            echo "seed: $i, datatypes: $types, data number: $num"
            python main.py --env_id $TASK --data_type $types --dataset_num $num --seed $i --device $device --difftype $difftype --marltype $marltype --critic_load_path $critic_load_path --diffusion_load_path $diffusion_load_path
        done
    done
done
