#!/bin/bash
#SBATCH --job-name=eegrecon
#SBATCH --ntasks=1
#SBATCH --output=/home/sanama/brain-diffuser_eeg/eegtest_%j.out
#SBATCH --error=/home/sanama/brain-diffuser_eeg/eegtest_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal_gpu
#SBATCH --chdir=/home/sanama/brain-diffuser_eeg/

# Make sure log dir exists
mkdir -p /home/sanama/brain-diffuser/logs

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

# echo "Running VDVAE Extract Features eeg"
# python3 test_eeg.py

echo "running vdvae recons"

python3 vdvae_recons_eeg.py \
  --pred_latents_path brain-diffuser/data/predicted_features/eeg_10test.npy \
  --compute_true_latents \
  --image_root EEG_dataset/things-eeg/Image_set/test_images/ \
  --num_images 10 \
  --batch_size 10 \
  --out_dir results/vdvae_out
