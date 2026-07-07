# 定义不同数据类型及其对应的 dataset_seed
# simple spread 数据集用 0 0 0 2
# simple tag 4 0 4 0
# simple world 1 0 1 2

# simple spread 新增测试 replay 3 random 3
# simple tag 新增测试 replay 0 random 1


TASK="simple_spread"   # simple xx
device=(1 2 3 4 5)
declare -A dataset_seeds=(
    ["expert"]=0
    ["medium"]=0
    ["medium-replay"]=0
    ["random"]=2
)

marltype="JAL"      # Can train JAL, IND    seq使用 jal critic
datatypes=("medium" "random" "expert") # "expert" "medium" "random"  
mix=False

# 检查 datatypes 数组中是否包含 "replay"
if [[ " ${datatypes[@]} " =~ " replay " ]]; then
    bl=62500
else
    bl=1000000
fi
# 已经不需要mediumreplay了
# if train on mixed data: --mixed_data

# Training Params
env_seed=(37)
training_epoch=500
training_steps_per_epoch=1000
save_interval=-1

# 调节IQL
taus=(0.5 0.7 0.9)   # default 0.7
temps=(0.5 1.0 3.0 5.0 7.0)   # default 3

# simple tag expert tau 0.5 0.7 0.9 + temp 7.0几乎都是最好的，其中0.7=0.5>0.9。250左右score
# simpel tag medium tau 0.5, temp 1.0>0.5>3.0，140左右score
# simple tag random tau 0.7 + temp 5.0/3.0 是衰减比较少的，0.5最差，0.9还可以，140左右score

# simple world random tau 0.7 + temp 7 5 3 不错  (0.7>0.5)
# simple world expert tau 0.5 + temp 3 1 5 不错  (0.7>0.5)   0.5&3.0 + 0.7&5.0
# simple world medium tau 0.7 + temp 3 1 5 不错  (0.7>0.5)   0.7&3.0/5.0  最好的是0.9&1.0

# simple spread expert tau 0.5 + temp 3 7   525-520
# simple spread medium tau 0.7 + temp 0.5   305-280
# simple spread random tau 0.5 + temp 0.5   240-220



# 计数器，用于轮流分配 GPU
counter=0

for types in "${datatypes[@]}"
do
    for tau in "${taus[@]}"
    do
        for temp in "${temps[@]}"
        do
            # 使用取模运算来选择GPU
            current_device=${device[$((counter % ${#device[@]}))]}
            
            seed=${dataset_seeds[$types]}
            echo "data types: $types, device: $current_device, data number: $seed, mix datasets: $mix"

            python /data/qiaodan/code/diffmarl/pretrain_critic.py \
                --env_id $TASK \
                --data_type $types \
                --dataset_num $seed \
                --device $current_device \
                --seed $env_seed\
                --training_epoch $training_epoch\
                --training_steps_per_epoch $training_steps_per_epoch\
                --save_interval $save_interval\
                --tau $tau \
                --temp $temp \
                --srpo_mode $marltype &
                
            # 增加计数器
            ((counter++))
        done
    done
done

wait