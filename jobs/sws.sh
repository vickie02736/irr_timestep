echo "This script is running on "
hostname

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate imae

python ../database/shallow_water/simulation.py
# python ../database/shallow_water/split_dataset.py

echo "All simulations completed successfully"