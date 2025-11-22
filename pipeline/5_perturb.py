#!/usr/bin/env python3
"""
JDM Disease Target Discovery using Geneformer In Silico Perturbation
Adapted for TNJDM vs HC cell classification
"""

from geneformer import InSilicoPerturber
from geneformer import InSilicoPerturberStats
from geneformer import EmbExtractor
import os
from datasets import load_from_disk
from multiprocessing import freeze_support
import datasets
from geneformer import perturber_utils

# Patch to fix compatibility with datasets >= 4.0.0
def patch_geneformer_column_fix():
    original_make_perturbation_batch_special = perturber_utils.make_perturbation_batch_special
    
    def patched_make_perturbation_batch_special(example_cell, perturb_type, tokens_to_perturb, anchor_token, combo_lvl, num_proc):
        # Force slicing for compatibility
        if int(datasets.__version__.split(".")[0]) >= 4:
            example_cell = example_cell[:]
        return original_make_perturbation_batch_special(example_cell, perturb_type, tokens_to_perturb, anchor_token, combo_lvl, num_proc)
    
    perturber_utils.make_perturbation_batch_special = patched_make_perturbation_batch_special
    print("Applied Geneformer Column object compatibility patch.")

patch_geneformer_column_fix()

def main():
    FINE_TUNED_MODEL_PATH = "savedmodels/trial_1da7fc18/250820_geneformer_cellClassifier_ray_trial_1da7fc18/ksplit1/checkpoint-125"
    INPUT_DATA_PATH = "tokenized_dataset/CD4All_JDM.dataset"
    OUTPUT_DIRECTORY = "jdm_perturbation_results"
    OUTPUT_PREFIX = "jdm_target_discovery"
    
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    MODEL_VERSION = "V2"
    NUM_CLASSES = 2
    # V2 uses 'cls' token embedding; V1 uses 'cell'
    EMB_MODE = "cls" if MODEL_VERSION == "V2" else "cell"

    N_PROCESSES = 4
    FORWARD_BATCH_SIZE = 16
    MAX_NCELLS = 1000
    
    print("Loading dataset...")
    dataset = load_from_disk(INPUT_DATA_PATH)
    
    # Define states: Disease (Start) -> Healthy (Goal)
    cell_states_to_model = {
        "state_key": "disease_group",
        "start_state": "TNJDM",
        "goal_state": "HC",
        "alt_states": []
    }
    
    filter_data_dict = {}

    state_embs_file = os.path.join(OUTPUT_DIRECTORY, f"{OUTPUT_PREFIX}.pkl")
    
    if os.path.exists(state_embs_file):
        print(f"Loading existing state embeddings from {state_embs_file}")
        import pickle
        with open(state_embs_file, 'rb') as f:
            state_embs_dict = pickle.load(f)
    else:
        print("Extracting new state embeddings...")
        embex = EmbExtractor(
            model_type="CellClassifier",
            num_classes=NUM_CLASSES,
            filter_data=filter_data_dict,
            max_ncells=MAX_NCELLS,
            emb_layer=0,
            summary_stat="exact_mean",
            forward_batch_size=FORWARD_BATCH_SIZE,
            model_version=MODEL_VERSION,
            nproc=N_PROCESSES
        )
        
        state_embs_dict = embex.get_state_embs(
            cell_states_to_model,
            FINE_TUNED_MODEL_PATH,
            INPUT_DATA_PATH,
            OUTPUT_DIRECTORY,
            OUTPUT_PREFIX
        )

    print("Initializing perturber...")
    isp = InSilicoPerturber(
        perturb_type="delete",
        perturb_rank_shift=None,
        genes_to_perturb="all",
        combos=0,
        anchor_gene=None,
        model_type="CellClassifier",
        num_classes=NUM_CLASSES,
        emb_mode=EMB_MODE,
        cell_emb_style="mean_pool",
        filter_data=filter_data_dict,
        cell_states_to_model=cell_states_to_model,
        state_embs_dict=state_embs_dict,
        max_ncells=MAX_NCELLS * 2,
        emb_layer=0,
        forward_batch_size=FORWARD_BATCH_SIZE,
        model_version=MODEL_VERSION,
        nproc=N_PROCESSES
    )
    
    isp.perturb_data(
        FINE_TUNED_MODEL_PATH,
        INPUT_DATA_PATH,
        OUTPUT_DIRECTORY,
        OUTPUT_PREFIX
    )

    print("Calculating perturbation statistics...")
    ispstats = InSilicoPerturberStats(
        mode="goal_state_shift",
        genes_perturbed="all",
        combos=0,
        anchor_gene=None,
        cell_states_to_model=cell_states_to_model,
        model_version=MODEL_VERSION
    )
    
    stats_output_dir = os.path.join(OUTPUT_DIRECTORY, "statistics")
    os.makedirs(stats_output_dir, exist_ok=True)
    
    ispstats.get_stats(
        OUTPUT_DIRECTORY,
        None,
        stats_output_dir,
        OUTPUT_PREFIX
    )

    try:
        import pandas as pd
        results_file = os.path.join(stats_output_dir, f"{OUTPUT_PREFIX}_stats_goal_state_shift.csv")
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            print("\nTop 10 candidate targets (highest goal_state_shift):")
            print(df.head(10)[['gene', 'goal_state_shift', 'pval_adj']].to_string(index=False))
    except ImportError:
        pass
    
    print(f"\nAnalysis complete. Results saved to: {OUTPUT_DIRECTORY}")

if __name__ == '__main__':
    freeze_support()
    main()
