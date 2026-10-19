#!/bin/bash
#SBATCH --job-name=fve_optuna
#SBATCH --output=logs/optuna_%j.out   
#SBATCH --error=logs/optuna_%j.err    
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                 
#SBATCH --cpus-per-task=1             
#SBATCH --mem=32G                     

source /lnet/aic/personal/hampejsj/aic_venv/bin/activate

mkdir -p logs
python -u train.py