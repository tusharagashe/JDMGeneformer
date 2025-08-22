#!/bin/bash
#SBATCH --job-name=perturb_geneformer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12      
#SBATCH --mem=64G             
#SBATCH --time=14:00:00        
#SBATCH --gres=gpu:1          
#SBATCH --output=%x-%j.out

# ⭐ Required for WandB to authenticate
export WANDB_API_KEY="b45ace5fda716efc07047beff449196357c3400e"
export WANDB_MODE="offline"


source .venv/bin/activate

nvidia-smi

# ⭐ Run the corrected Python script
python pipeline/5_perturb.py



# ython pipeline/5_perturb.py 
# Checking and fixing dataset format...
# Original dataset size: 41455
# Sample input_ids type: <class 'list'>
# Fixed dataset saved to: tokenized_dataset/CD4All_JDM_perturbation_ready.dataset
# Fixed input_ids type: <class 'list'>
# Setting up JDM perturbation analysis...
# Model version: V2
# Embedding mode: cls
# Cell states configuration: {'state_key': 'disease_group', 'start_state': 'TNJDM', 'goal_state': 'HC', 'alt_states': []}

# Step 1: Loading existing state embeddings from jdm_perturbation_results/jdm_target_discovery.pkl
# State embeddings loaded successfully!
# Found states: ['TNJDM', 'HC']

# Step 2: Setting up in silico perturbation...
# In silico perturber initialized!

# Step 3: Running perturbation analysis...
# This may take a while depending on dataset size and number of genes...
#   0%|                                                                                                                                             | 0/2000 [00:00<?, ?it/s/c4/home/tagashe/JDMGeneformer/.venv/lib64/python3.11/site-packages/torch/nn/modules/module.py:1762: FutureWarning: `encoder_attention_mask` is deprecated and will be removed in version 4.55.0 for `BertSdpaSelfAttention.forward`.
#   return forward_call(*args, **kwargs)
#   0%|                                                                                                                                             | 0/2000 [00:00<?, ?it/s]
# Traceback (most recent call last):                                                                                                                                         
#   File "/c4/home/tagashe/JDMGeneformer/pipeline/5_perturb.py", line 256, in <module>
#     main()
#   File "/c4/home/tagashe/JDMGeneformer/pipeline/5_perturb.py", line 175, in main
#     isp.perturb_data(
#   File "/c4/home/tagashe/JDMGeneformer/.venv/lib64/python3.11/site-packages/geneformer/in_silico_perturber.py", line 508, in perturb_data
#     self.isp_perturb_all_special(
#   File "/c4/home/tagashe/JDMGeneformer/.venv/lib64/python3.11/site-packages/geneformer/in_silico_perturber.py", line 1350, in isp_perturb_all_special
#     perturbation_batch, indices_to_perturb = pu.make_perturbation_batch_special(
#                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/c4/home/tagashe/JDMGeneformer/.venv/lib64/python3.11/site-packages/geneformer/perturber_utils.py", line 565, in make_perturbation_batch_special
#     "input_ids": example_cell["input_ids"] * length,
#                  ~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~
# TypeError: unsupported operand type(s) for *: 'Column' and 'int'