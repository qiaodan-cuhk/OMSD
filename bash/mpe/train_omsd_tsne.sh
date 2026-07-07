# 用于收集eval policy数据，并进行t-sne可视化
# simple spread 数据集用 0 0 0 2
# simple tag 4 0 4 0
# simple world 1 0 1 2

# 每次eval的时候记录了buffer数据

#!/bin/sh

# 定义任务列表
tasks=("simple_spread")  # "simple_spread" 

# 根据不同任务和数据类型设置对应的dataset_num
declare -A dataset_seeds=(
    # simple spread的数据集编号
    ["expert"]=0
    ["medium"]=0
    ["random"]=2
    # # simple tag的数据集编号
    # ["expert"]=4
    # ["medium"]=0
    # ["random"]=0
    # # simple world的数据集编号
    # ["expert"]=1
    # ["medium"]=0
    # ["random"]=2
)

seeds=(42)  # 可以根据需要添加更多种子
device=(1 2 3)  # 可用的GPU设备
datatypes=("medium" "random")
diffusion_epoch_num=(199)
critic_epoch_num=(499)

difftype="SRPO"
marltype="SEQ"
counter=0
trainsteps=100000
annealing_epochs=10

for TASK in "${tasks[@]}"
do
    for types in "${datatypes[@]}"
    do
        # 在循环内部根据任务类型和数据类型设置beta
        if [ "$types" == "random" ]; then
            if [ "$TASK" == "simple_spread" ]; then
                beta=(0.05)
            elif [ "$TASK" == "simple_tag" ]; then
                beta=(0.5)
            elif [ "$TASK" == "simple_world" ]; then
                beta=(0.5)
            fi
        elif [ "$types" == "expert" ]; then 
            if [ "$TASK" == "simple_spread" ]; then
                beta=(0.001)
            elif [ "$TASK" == "simple_tag" ]; then
                beta=(0.005)
            elif [ "$TASK" == "simple_world" ]; then
                beta=(0.01)
            fi
        elif [ "$types" == "medium" ]; then
            if [ "$TASK" == "simple_spread" ]; then
                beta=(0.005)
            elif [ "$TASK" == "simple_tag" ]; then
                beta=(0.05)
            elif [ "$TASK" == "simple_world" ]; then
                beta=(0.05)
            fi
        else
            beta=(0.1)
        fi

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
                        
                        echo "task: $TASK, seed: $i, datatypes: $types, data number: $dataset_num"
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

                        counter=$((counter + 1))
                    done
                done
            done
        done
    done
done

wait