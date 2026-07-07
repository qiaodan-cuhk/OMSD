#!/bin/sh
seeds=(100 37 888)
datatypes=("expert")  # "expert" "random" "medium" "medium-replay"
datanums=0    # 1 2 3
beta=(0.001 0.002 0.005 0.01)

## algo seletion
difftype="SRPO"     # DQL
marltype="IND"      # JAL, VD, SEQ
TASK="simple_spread"   # simple xx
device=0


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
            python main.py --env_id $TASK --data_type $types --dataset_num $num --seed $i --device $device --difftype $difftype --marltype $marltype --critic_load_path $critic_load_path --diffusion_load_path $diffusion_load_path --beta $beta
        done
    done
done
