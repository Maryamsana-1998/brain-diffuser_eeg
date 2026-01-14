#!/bin/bash
#SBATCH --job-name=eegfeatures
#SBATCH --ntasks=1
#SBATCH --output=/home/sanama/brain-diffuser_eeg/logs/eegtest_%j.out
#SBATCH --error=/home/sanama/brain-diffuser_eeg/logs/eegtest_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal_gpu
#SBATCH --chdir=/home/sanama/brain-diffuser_eeg/


# Debugging output
echo "==== Job started on $(hostname) at $(date) ===="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"
echo "SLURM_NTASKS: $SLURM_NTASKS"

# Load Conda properly
module purge
module load miniconda
echo "Loaded miniconda"

source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Activated Conda environment: $(which python)"
python -c "import torch; print('cuda available:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"

echo " Running eeg vision and text feature extraction"

echo " TEXT EXTRACTION"
python3 cliptext_features_eeg.py -sub 1 

echo " VISION EXTRACTION"
python3 clipvision_features_eeg.py -sub 1