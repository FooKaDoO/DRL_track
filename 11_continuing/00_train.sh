#!/bin/bash
#SBATCH -J tetris
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 100:00:00

source ../.venv/bin/activate

python3 -B train.py
