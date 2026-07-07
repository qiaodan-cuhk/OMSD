# 添加了IQL的grad clip重新训练

# 定义不同任务及其对应的 dataset_seed
declare -A TASKS=(
    # ["simple_spread"]="medium"   # 试试正常训练的critic加了clamp怎么样
    # ["simple_spread"]="medium-replay"
    ["simple_tag"]="medium expert medium-replay"
    # ["simple_world"]="medium-replay expert"
)

declare -A dataset_seeds=(
    # simple spread seeds
    ["simple_spread,expert"]=0
    ["simple_spread,medium"]=0
    ["simple_spread,medium-replay"]=0
    ["simple_spread,random"]=2

    # simple tag seeds  
    ["simple_tag,expert"]=4
    ["simple_tag,medium"]=0
    ["simple_tag,medium-replay"]=4
    ["simple_tag,random"]=0
    

    # simple world seeds
    ["simple_world,medium-replay"]=1
    ["simple_world,expert"]=1
)

batch_size=(512)
critic_lr=3e-4

device=(4 5)
training_seeds=(42)
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
        dataset_seed=${dataset_seeds["$task,$type"]}

        for bs in "${batch_size[@]}"
        do
            for training_seed in "${training_seeds[@]}"
            do 
                echo "task: $task, data type: $type, device: $current_device, training seed: $training_seed"
                echo "data number: $dataset_seed, mix datasets: $mix, iql_lr: $critic_lr, batch size: $bs"
                python /data/qiaodan/code/diffmarl/pretrain_critic.py \
                    --env_id $task \
                    --data_type $type \
                    --dataset_num $dataset_seed \
                    --device $current_device \
                    --srpo_mode $marltype \
                    --batch_size $bs \
                    --iql_critic_lr $critic_lr \
                    --seed $training_seed &
                    
                # 增加计数器
                ((counter++))
            done
        done
    done
done

wait