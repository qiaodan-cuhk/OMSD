#!/bin/sh
# seeds=(42 99 1000 9999 23477)   #  100
seeds=100


types=("expert") # "expert" "random" "medium-replay"
datanums=3    # 1 2 3

## algo seletion
difftype="SRPO"     # DQL
marltype="SEQ"      
TASK="HalfCheetah-v2"
device=3

# beta=(0.001 0.005 0.01 0.02 0.05 0.1 0.2 0.5)  # medium-replay 在0.02附近表现较好

# beta=(0.01 0.015 0.02 0.025 0.03 0.04)
beta=(0.01 0.02 0.05 0.1 0.2 0.5)



# 0.05 - 0.25附近 for random
# beta=(0.02)
# beta=(0.0001 0.001 0.01 0.02 0.05 0.1 0.2 0.5 1)




diffusion_epoch_num=149
critic_epoch_num=79


trainsteps=1000000

for i in "${seeds[@]}"
do
    for beta_i in "${beta[@]}"
    do
        # actor/critic load path
        critic_load_path="/data/qiaodan/code/diffmarl/SRPO_premodels/HalfCheetah-v2"
        diffusion_load_path="/data/qiaodan/code/diffmarl/SRPO_premodels/HalfCheetah-v2"


        # 打印路径以进行调试
        echo "Critic Load Path: $critic_load_path"
        echo "Diffusion Load Path: $diffusion_load_path"

        echo "seed: $i, datatypes: $types, data number: $datanums, beta: $beta_i"
        python main.py --env_id $TASK --data_type $types --dataset_num $datanums --seed $i --device $device --difftype $difftype --marltype $marltype --critic_load_path $critic_load_path --diffusion_load_path $diffusion_load_path --diff_epoch $diffusion_epoch_num --critic_epoch $critic_epoch_num --num_steps $trainsteps --beta $beta_i &

    done
done

# 等待所有后台任务完成
wait
