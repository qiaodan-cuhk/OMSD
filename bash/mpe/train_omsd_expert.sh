#!/bin/sh
# simple spread 数据集用 0 0 0 2
# simple tag 4 0 4 0
# simple world 1 0 1 2

TASKS=("simple_world")   # "simple_tag" "simple_world"
datatypes=("expert" "medium" "random")  # 添加多个数据类型

# 修改为二维关联数组，为每个task和datatype组合指定beta值
declare -A type_betas
# simple_spread的不同数据类型对应的beta值
type_betas["simple_spread,expert"]="0.02 0.1 0.2 0.5"
type_betas["simple_spread,medium"]="0.001 0.1 0.2 0.5"
type_betas["simple_spread,random"]="0.001 0.005 0.01 0.02 0.05 0.1 0.2"

# simple_tag的不同数据类型对应的beta值
type_betas["simple_tag,expert"]="0.005 0.01"
type_betas["simple_tag,medium"]="0.015 0.02"
type_betas["simple_tag,random"]="0.025 0.03"

# simple_world的不同数据类型对应的beta值
type_betas["simple_world,expert"]="0.1 0.2 0.5"
type_betas["simple_world,medium"]="0.001 0.005 0.1 0.2 0.5"
type_betas["simple_world,random"]="0.001 0.005 0.01 0.02 0.05 0.1"

# 修改为二维关联数组，为每个task和datatype组合指定dataset seed
declare -A dataset_seeds
# simple_spread的不同数据类型对应的dataset seed
dataset_seeds["simple_spread,expert"]="0"
dataset_seeds["simple_spread,medium"]="0"
dataset_seeds["simple_spread,random"]="2"

# simple_tag的不同数据类型对应的dataset seed
dataset_seeds["simple_tag,expert"]="4"
dataset_seeds["simple_tag,medium"]="0"
dataset_seeds["simple_tag,random"]="0"

# simple_world的不同数据类型对应的dataset seed
dataset_seeds["simple_world,expert"]="1"
dataset_seeds["simple_world,medium"]="0"
dataset_seeds["simple_world,random"]="2"

seeds=(42)
device=(1 2 3 4 5)

diffusion_epoch_num=(199)           # 149
critic_epoch_num=(119)    # 99

difftype="SRPO"
marltype="SEQ"
counter=0
trainsteps=100000
annealing_epochs=10

for TASK in "${TASKS[@]}"
do
    for datatype in "${datatypes[@]}"
    do
        for i in "${seeds[@]}"
        do
            for critic_i in "${critic_epoch_num[@]}"
            do
                for diff_i in "${diffusion_epoch_num[@]}"
                do
                    # 使用task,datatype组合作为键获取对应的beta值和dataset seed
                    key="${TASK},${datatype}"
                    for beta_i in ${type_betas[$key]}
                    do
                        dataset_num=${dataset_seeds[$key]}
                        current_device=${device[$((counter % ${#device[@]}))]}

                        echo "task: $TASK, data type: $datatype, seed: $i, dataset_num: $dataset_num"
                        echo "beta: $beta_i, device: $current_device, critic: $critic_i, diff: $diff_i"
                        
                        python main.py --env_id $TASK \
                                    --data_type $datatype \
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
done

wait