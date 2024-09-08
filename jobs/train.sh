#!/bin/bash

#SBATCH --job-name=imae_wt          # 作业名称
#SBATCH --account=PAS2490		    # Project ID
#SBATCH --nodes=1                   # 节点数
#SBATCH --ntasks-per-node=1         # 每个节点的任务数
#SBATCH --cpus-per-task=3           # 每个任务使用的 CPU 核心数
#SBATCH --gpus-per-node=1           # GPU per node
#SBATCH --mem=200G                   # 内存限制
#SBATCH --time=24:00:00             # 作业运行时间限制

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=uceckz0@ucl.ac.uk


source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate imae


convert_to_seconds() {
    date -d "${1//_/ }" +%s
}


start_time=$(date +%Y-%m-%d_%H:%M:%S)
echo "Train start time: $start_time"
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=101 --rdzv_endpoint=localhost:29505 ../program/main.py\
        --cpu 3\
        --epochs 500\
        --resume-epoch 1\
        --database weather_2m_temperature\
        --save-frequency 10\
        --model-name imae

# model-name: imae, convlstm, cae, cae_lstm 
# database: shallow_water, weather_2m_temperature
# interpolation: linear, gaussian

end_time=$(date +%Y-%m-%d_%H:%M:%S)
echo "Train end time: $end_time"


# start_time=$(date +%Y-%m-%d_%H:%M:%S)
# echo "Test start time: $start_time"

# for ratio in $(seq 0.1 0.1 0.9); do
#     torchrun --nnodes=1 --nproc_per_node=1 ../program/main.py \
#              --test-flag True \
#              --resume-epoch 601 \
#              --database shallow_water \
#              --mask-ratio $ratio \
#              --model-name imae
# done

# end_time=$(date +%Y-%m-%d_%H:%M:%S)
# echo "Test end time: $end_time"



# start_seconds=$(convert_to_seconds "$start_time")
# end_seconds=$(convert_to_seconds "$end_time")

# difference_seconds=$((end_seconds - start_seconds))
# hours=$((difference_seconds / 3600))
# minutes=$(( (difference_seconds % 3600) / 60 ))
# seconds=$((difference_seconds % 60))

 echo "Test Time taken: $hours hours, $minutes minutes and $seconds seconds"
