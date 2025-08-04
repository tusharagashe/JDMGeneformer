#!/bin/bash
#SBATCH --job-name=finetune_geneformer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      
#SBATCH --mem=64G             
#SBATCH --time=12:00:00        
#SBATCH --gres=gpu:1          
#SBATCH --partition=gpu        
#SBATCH --output=%x-%j.out


source .venv/bin/activate

nvidia-smi

python  pipeline/2_finetune.py 