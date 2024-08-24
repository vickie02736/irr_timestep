source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate imae

start_time=$(date +%Y-%m-%d_%H:%M:%S)
echo "start time: $start_time"

# for ratio in $(seq 0.1 0.1 0.9); do
#     torchrun --nnodes=1 --nproc_per_node=1 ../program/main.py \
#              --test-flag True \
#              --resume-epoch 5 \
#              --database shallow_water \
#              --mask-ratio $ratio \
#              --model-name imae
# done



torchrun --nnodes=1 --nproc_per_node=1 ../program/main.py \
            --test-flag True \
            --resume-epoch 5 \
            --database shallow_water \
            --mask-ratio 0.1 \
            --model-name imae

# model name: imae, convlstm, cae, cae_lstm 

end_time=$(date +%Y-%m-%d_%H:%M:%S)
echo "end time: $end_time"

convert_to_seconds() {
    date -d "${1//_/ }" +%s
}


start_seconds=$(convert_to_seconds "$start_time")
end_seconds=$(convert_to_seconds "$end_time")

difference_seconds=$((end_seconds - start_seconds))
hours=$((difference_seconds / 3600))
minutes=$(( (difference_seconds % 3600) / 60 ))
seconds=$((difference_seconds % 60))

echo "Time taken: $hours hours, $minutes minutes and $seconds seconds"