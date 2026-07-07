#!/bin/sh
seeds=(100 37 888)
datatypes=("expert")  # "expert" "random" "medium" "medium-replay"
datanums=0    # 1 2 3
beta=(0.001 0.002 0.005 0.01)

## algo seletion
difftype="SRPO"     # DQL
marltype="IND"      # JAL, VD, SEQ
TASK="HalfCheetah-v2"
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


# --------------------------------------
# random ind SRPO的训练超参数
# beta=(0.15 0.17 0.19 0.21 0.23 0.25)
# beta=(0.0005 0.07 0.15 0.3 0.7 2 3 10)
# beta=(0.002)
# random需要0.2附近
# diffusion_epoch_num=149
# critic_epoch_num=119