#!/bin/bash
#SBATCH --job-name "DQN Pong"
#SBATCH --output "log.out"
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --mem=5GB

source /home/tullwd25/691/Atari-Pong-Deep-Q-Network/.venv/bin/activate
python DQN.py