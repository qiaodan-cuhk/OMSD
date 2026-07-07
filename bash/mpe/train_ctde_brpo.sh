#!/bin/sh
seeds=(42 88 379)
datatypes=("expert")  # "expert" "random" "medium-replay" "medium"
datanums=0    # 1 2 3

## algo seletion
difftype="SRPO"     # DQL
marltype="CTDE"      # JAL, VD, SEQ
TASK="simple_spread"   # simple xx
device=1
beta=0.001


for i in "${seeds[@]}"
do
    for types in "${datatypes[@]}"
    do
        # actor/critic load path
        critic_load_path="/home/qiaodan/Code/diffmarl/SRPO_premodels/HalfCheetah-v2"
        diffusion_load_path="/home/qiaodan/Code/diffmarl/SRPO_premodels/HalfCheetah-v2"

        for num in "${datanums[@]}"
        do
            echo "seed: $i, datatypes: $types, data number: $num"
            python main.py --env_id $TASK --data_type $types --dataset_num $num --seed $i --device $device --difftype $difftype --marltype $marltype --critic_load_path $critic_load_path --diffusion_load_path $diffusion_load_path --beta $beta &
        done
    done
done

