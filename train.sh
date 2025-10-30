#!/bin/bash
#SBATCH --job-name "DQN Pong"
#SBATCH --output "log_DQN.out"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=8GB

module load apps/python/3.11.8
source /home/tullwd25/691/Atari-Pong-Deep-Q-Network/.venv/bin/activate
python run.py