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
from multiprocessing import freeze_support # Import freeze_support

def main():
    # =============================================================================
    # CONFIGURATION - UPDATE THESE PATHS FOR YOUR PROJECT
    # =============================================================================
    
    
    # Model and data paths (UPDATE THESE)
    FINE_TUNED_MODEL_PATH = "savedmodels/trial_1da7fc18/250820_geneformer_cellClassifier_ray_trial_1da7fc18/ksplit1/checkpoint-125"  # Your fine-tuned JDM model
    INPUT_DATA_PATH = "tokenized_dataset/CD4All_JDM.dataset"        # Your test set in .dataset format
    OUTPUT_DIRECTORY = "jdm_perturbation_results"       # Where to save results
    OUTPUT_PREFIX = "jdm_target_discovery"                      # Prefix for output files

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)  # Create output directory if it doesn't exist

    # Model configuration (UPDATE IF DIFFERENT)
    MODEL_VERSION = "V2"  # Update to match your model version (V1 or V2)
    NUM_CLASSES = 2       # TNJDM vs HC = 2 classes
    EMB_MODE = "cls" if MODEL_VERSION == "V2" else "cell"  # cls for V2, cell for V1

    # Computational resources (ADJUST AS NEEDED)
    N_PROCESSES = 4       # Number of processes for parallel processing
    FORWARD_BATCH_SIZE = 16  # Batch size for forward passes
    MAX_NCELLS = 1000     # Maximum number of cells to analyze per state
    
    # =============================================================================
    # FIX DATASET FORMAT (for perturbation compatibility)
    # =============================================================================
    
    print("Checking and fixing dataset format...")
    
    # Load dataset and check format
    dataset = load_from_disk(INPUT_DATA_PATH)
    print(f"Original dataset size: {len(dataset)}")
    print(f"Sample input_ids type: {type(dataset[0]['input_ids'])}")
    
    # Fix input_ids format if needed
    def fix_input_ids_format(example):
        # Convert Column/tensor to list for perturbation compatibility
        if hasattr(example["input_ids"], 'to_list'):
            example["input_ids"] = example["input_ids"].to_list()
        elif hasattr(example["input_ids"], 'tolist'):
            example["input_ids"] = example["input_ids"].tolist()
        elif not isinstance(example["input_ids"], list):
            example["input_ids"] = list(example["input_ids"])
        return example
    
    # This is critical for preventing the TypeError in child processes
    dataset = dataset.map(fix_input_ids_format, num_proc=N_PROCESSES)
    
    # Save fixed dataset temporarily
    fixed_dataset_path = INPUT_DATA_PATH.replace(".dataset", "_perturbation_ready.dataset")
    dataset.save_to_disk(fixed_dataset_path)
    INPUT_DATA_PATH = fixed_dataset_path
    
    print(f"Fixed dataset saved to: {fixed_dataset_path}")
    print(f"Fixed input_ids type: {type(dataset[0]['input_ids'])}")
    
    # =============================================================================
    # STEP 1: DEFINE CELL STATES FOR JDM ANALYSIS
    # =============================================================================
    
    print("Setting up JDM perturbation analysis...")
    print(f"Model version: {MODEL_VERSION}")
    print(f"Embedding mode: {EMB_MODE}")
    
    # Define the cell states for your JDM analysis
    cell_states_to_model = {
        "state_key": "disease_group",  # Update this to match your metadata column name
        "start_state": "TNJDM",    # Disease state (starting point)
        "goal_state": "HC",        # Healthy control state (goal)
        "alt_states": []          # No alternative states for binary classification
    }
    
    # Optional: Filter data by specific cell types if you have multiple
    # Leave empty dict {} if you want to analyze all cells
    filter_data_dict = {}  # e.g., {"tissue": ["muscle"], "batch": ["batch1"]}
    
    print(f"Cell states configuration: {cell_states_to_model}")
    
    # =============================================================================
    # STEP 2: LOAD OR EXTRACT STATE EMBEDDINGS
    # =============================================================================
    
    # Check if state embeddings already exist
    state_embs_file = os.path.join(OUTPUT_DIRECTORY, f"{OUTPUT_PREFIX}.pkl")
    
    if os.path.exists(state_embs_file):
        print(f"\nStep 1: Loading existing state embeddings from {state_embs_file}")
        import pickle
        with open(state_embs_file, 'rb') as f:
            state_embs_dict = pickle.load(f)
        print("State embeddings loaded successfully!")
        print(f"Found states: {list(state_embs_dict.keys())}")
        
    else:
        print("\nStep 1: Extracting new state embeddings...")
        
        # Initialize embedding extractor
        embex = EmbExtractor(
            model_type="CellClassifier",      # Using your fine-tuned classifier
            num_classes=NUM_CLASSES,          # TNJDM vs HC = 2 classes
            filter_data=filter_data_dict,     # Any data filtering
            max_ncells=MAX_NCELLS,            # Max cells per state to analyze
            emb_layer=0,                      # Layer to extract embeddings from
            summary_stat="exact_mean",        # How to summarize cell embeddings
            forward_batch_size=FORWARD_BATCH_SIZE,
            model_version=MODEL_VERSION,
            nproc=N_PROCESSES
        )
        
        # Extract embeddings for TNJDM and HC states
        state_embs_dict = embex.get_state_embs(
            cell_states_to_model,
            FINE_TUNED_MODEL_PATH,
            INPUT_DATA_PATH,
            OUTPUT_DIRECTORY,
            OUTPUT_PREFIX
        )
        
        print("State embeddings extracted and saved successfully!")
    
    # =============================================================================
    # STEP 3: SETUP IN SILICO PERTURBER
    # =============================================================================
    
    print("\nStep 2: Setting up in silico perturbation...")
    
    # Initialize the perturber for gene deletion analysis
    isp = InSilicoPerturber(
        perturb_type="delete",             # Delete genes to see effect
        perturb_rank_shift=None,           # Not applicable for deletion
        genes_to_perturb="all",            # Test all genes (can specify list if needed)
        combos=0,                          # Single gene perturbations only
        anchor_gene=None,                  # No anchor gene
        model_type="CellClassifier",       # Using your fine-tuned classifier
        num_classes=NUM_CLASSES,           # TNJDM vs HC = 2 classes
        emb_mode=EMB_MODE,                 # "cls" for V2, "cell" for V1
        cell_emb_style="mean_pool",        # How to pool cell embeddings
        filter_data=filter_data_dict,      # Any data filtering
        cell_states_to_model=cell_states_to_model,
        state_embs_dict=state_embs_dict,   # Previously extracted embeddings
        max_ncells=MAX_NCELLS * 2,         # Slightly higher for perturbation
        emb_layer=0,                       # Same layer as embedding extraction
        forward_batch_size=FORWARD_BATCH_SIZE,
        model_version=MODEL_VERSION,
        nproc=N_PROCESSES
    )
    
    print("In silico perturber initialized!")
    
    # =============================================================================
    # STEP 4: RUN PERTURBATION ANALYSIS
    # =============================================================================
    
    print("\nStep 3: Running perturbation analysis...")
    print("This may take a while depending on dataset size and number of genes...")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # Run the perturbation analysis
    isp.perturb_data(
        FINE_TUNED_MODEL_PATH,
        INPUT_DATA_PATH,
        OUTPUT_DIRECTORY,
        OUTPUT_PREFIX
    )
    
    print("Perturbation analysis completed!")
    
    # =============================================================================
    # STEP 5: CALCULATE STATISTICS
    # =============================================================================
    
    print("\nStep 4: Calculating perturbation statistics...")
    
    # Initialize statistics calculator
    ispstats = InSilicoPerturberStats(
        mode="goal_state_shift",             # Analyze shifts toward goal state (HC)
        genes_perturbed="all",               # All genes were perturbed
        combos=0,                            # Single gene perturbations
        anchor_gene=None,                    # No anchor gene
        cell_states_to_model=cell_states_to_model,
        model_version=MODEL_VERSION
    )
    
    # Calculate and save statistics
    stats_output_dir = os.path.join(OUTPUT_DIRECTORY, "statistics")
    os.makedirs(stats_output_dir, exist_ok=True)
    
    ispstats.get_stats(
        OUTPUT_DIRECTORY,                    # Directory with perturbation results
        None,                                # No additional filtering
        stats_output_dir,                    # Where to save statistics
        OUTPUT_PREFIX
    )
    
    print("Statistics calculation completed!")
    
    # =============================================================================
    # RESULTS SUMMARY
    # =============================================================================
    
    print(f"""
=============================================================================
JDM TARGET DISCOVERY ANALYSIS COMPLETE!
=============================================================================
    
Results saved to: {OUTPUT_DIRECTORY}
Statistics saved to: {stats_output_dir}
    
Key output files:
- {OUTPUT_PREFIX}_stats_goal_state_shift.csv: Main results with gene rankings
- {OUTPUT_PREFIX}_stats_summary.csv: Summary statistics
    
Next steps:
1. Examine the CSV files to identify top gene targets
2. Look for genes with high goal_state_shift values (TNJDM → HC)
3. Cross-reference with known JDM pathways and literature
4. Consider validation experiments for top candidates
    
The goal_state_shift metric indicates how effectively deleting each gene
pushes TNJDM cells toward a healthy control (HC) phenotype.
=============================================================================
""")
    
    # Optional: Quick preview of results if pandas is available
    try:
        import pandas as pd
        results_file = os.path.join(stats_output_dir, f"{OUTPUT_PREFIX}_stats_goal_state_shift.csv")
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            print("Top 10 candidate targets (highest goal_state_shift):")
            print(df.head(10)[['gene', 'goal_state_shift', 'pval_adj']].to_string(index=False))
    except ImportError:
        print("Install pandas to see a quick preview of top results")
    
    print("\nAnalysis completed successfully!")


if __name__ == '__main__':
    freeze_support()  # Required for multiprocessing on Windows/some systems
    main()
