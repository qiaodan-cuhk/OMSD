# 添加了IQL的grad clip重新训练

# 定义不同任务及其对应的 dataset_seed
declare -A TASKS=(
    ["simple_spread"]="medium-replay"
    ["simple_tag"]="medium-replay expert"
    ["simple_world"]="medium-replay expert"
)

declare -A dataset_seeds=(
    # simple spread seeds
    ["simple_spread,medium-replay"]=0

    # simple tag seeds  
    ["simple_tag,medium-replay"]=4
    ["simple_tag,expert"]=4

    # simple world seeds
    ["simple_world,medium-replay"]=1
    ["simple_world,expert"]=1
)

device=(5)
marltype="JAL"      # Can train JAL, IND    seq使用 jal critic
mix=False
# if train on mixed data: --mixed_data

# 计数器，用于轮流分配 GPU
counter=0

# 遍历每个任务
for task in "${!TASKS[@]}"
do
    # 获取该任务需要的数据类型
    datatypes=(${TASKS[$task]})
    
    # 遍历该任务的每个数据类型
    for type in "${datatypes[@]}"
    do
        # 使用取模运算来选择GPU
        current_device=${device[$((counter % ${#device[@]}))]}
        
        # 获取对应的 seed
        seed=${dataset_seeds["$task,$type"]}
        
        echo "task: $task, data type: $type, device: $current_device, data number: $seed, mix datasets: $mix"

        python /data/qiaodan/code/diffmarl/pretrain_critic.py \
            --env_id $task \
            --data_type $type \
            --dataset_num $seed \
            --device $current_device \
            --srpo_mode $marltype &
            
        # 增加计数器
        ((counter++))
    done
done

wait