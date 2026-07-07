#!/bin/bash
# 用于训练critic的脚本，使用最优参数

# 定义最优参数字典
declare -A optimal_params=(
    ["simple_tag,expert"]="0.7,7.0"
    ["simple_tag,medium"]="0.5,1.0"
    ["simple_tag,random"]="0.7,5.0"
    ["simple_world,expert"]="0.5,3.0"
    ["simple_world,medium"]="0.9,1.0"
    ["simple_world,random"]="0.7,7.0"
    ["simple_spread,expert"]="0.5,3.0"
    ["simple_spread,medium"]="0.7,0.5"
    ["simple_spread,random"]="0.5,0.5"
)

tasks=("simple_tag" "simple_world" "simple_spread")
device=(1 2 3 4 5)

# 定义每个任务的数据集种子
declare -A task_dataset_seeds=(
    ["simple_spread,expert"]=0
    ["simple_spread,medium"]=0
    ["simple_spread,random"]=2
    ["simple_tag,expert"]=4
    ["simple_tag,medium"]=0
    ["simple_tag,random"]=0
    ["simple_world,expert"]=1
    ["simple_world,medium"]=0
    ["simple_world,random"]=2
)

marltype="JAL"      # Can train JAL, IND
datatypes=("medium" "random" "expert")
mix=False

# 检查 datatypes 数组中是否包含 "replay"
if [[ " ${datatypes[@]} " =~ " replay " ]]; then
    bl=62500
else
    bl=1000000
fi

# Training Params
env_seed=(37)
training_epoch=500
training_steps_per_epoch=1000
save_interval=100

# 计数器，用于轮流分配 GPU
counter=0

for TASK in "${tasks[@]}"
do
    for types in "${datatypes[@]}"
    do
        # 从字典中获取对应的最优参数
        key="$TASK,$types"
        IFS=',' read -r tau temp <<< "${optimal_params[$key]}"
        
        # 使用取模运算来选择GPU
        current_device=${device[$((counter % ${#device[@]}))]}
        
        # 使用正确的变量名获取种子
        seed=${task_dataset_seeds["$TASK,$types"]}
        echo "Running experiment for $TASK, $types with tau=$tau, temp=$temp"
        echo "Device: $current_device, data number: $seed, mix datasets: $mix"
        echo "--------------------------------------------"

        python /data/qiaodan/code/diffmarl/pretrain_critic.py \
            --env_id $TASK \
            --data_type $types \
            --dataset_num $seed \
            --device $current_device \
            --seed $env_seed \
            --training_epoch $training_epoch \
            --training_steps_per_epoch $training_steps_per_epoch \
            --save_interval $save_interval \
            --tau $tau \
            --temp $temp \
            --srpo_mode $marltype &
                
        # 增加计数器
        ((counter++))
        
    done
done

# 等待所有后台任务完成
wait