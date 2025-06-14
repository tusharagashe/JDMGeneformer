#!/bin/bash
#SBATCH --job-name=finetune_geneformer_JDM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4      
#SBATCH --mem=64G             
#SBATCH --time=16:00:00        
#SBATCH --gres=gpu:1          
#SBATCH --partition=gpu        
#SBATCH --output=%x-%j.out


source ~/jdmanalysis/bin/activate

python ./geneformer_pipeline.py