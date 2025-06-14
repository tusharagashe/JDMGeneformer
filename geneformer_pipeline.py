#!/usr/bin/env python3
"""
Complete Geneformer Pipeline: scRNA-seq to Perturbation Analysis
CORRECTED VERSION - Using Latest Geneformer API (v0.1.0, 2024)
From raw data tokenization through fine-tuning to therapeutic target discovery
"""

import os
import pandas as pd
import scanpy as sc
from datasets import load_from_disk
from geneformer import TranscriptomeTokenizer, Classifier, InSilicoPerturber, InSilicoPerturberStats
from transformers import TrainingArguments

# ============================================================================
# STEP 1: TOKENIZE RAW scRNA-seq DATA
# ============================================================================

def tokenize_data(adata_path, output_dir="tokenized_dataset"):
    """
    Tokenize scRNA-seq data for Geneformer
    
    Args:
        adata_path: Path to AnnData object (.h5ad file)
        output_dir: Directory to save tokenized data
    """
    print("=== STEP 1: TOKENIZING DATA ===")
    
    # Create directories if they don't exist
    raw_data_dir = "raw_data"
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load your scRNA-seq data
    adata = sc.read_h5ad(adata_path)
    
    # Save to raw_data directory for tokenizer (it expects files in a directory)
    raw_file_path = os.path.join(raw_data_dir, "data.h5ad")
    adata.write_h5ad(raw_file_path)
    
    # Ensure proper format for tokenizer
    # Assumes you have:
    # - adata.obs['disease_group'] with values: ['Active', 'TNJDM', 'Inactive', 'HC']
    # - adata.var.index contains gene symbols or Ensembl IDs
    # - adata.X contains raw counts
    
    # Convert to format expected by TranscriptomeTokenizer
    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict={
            "disease_group": "disease_group",
            "cell_type": "cell_type"  # if you have this
        },
        nproc=4  # adjust based on your system
    )
    
    # Tokenize the dataset
    tokenizer.tokenize_data(
        data_directory=raw_data_dir,     # directory containing your .h5ad files
        output_directory=output_dir,
        output_prefix="tokenized"
    )
    
    print(f"Tokenized data saved to {output_dir}")
    return f"{output_dir}/tokenized.dataset"

# ============================================================================
# STEP 2: SPLIT DATA INTO TRAIN/TEST
# ============================================================================

def split_dataset(tokenized_path, train_ratio=0.8):
    """
    Split tokenized dataset into train and test sets
    """
    print("=== STEP 2: SPLITTING DATASET ===")
    
    # Load tokenized dataset
    dataset = load_from_disk(tokenized_path)
    
    # Split dataset
    train_test = dataset.train_test_split(test_size=1-train_ratio, seed=42)
    
    # Save splits
    train_test["train"].save_to_disk("tokenized_train_dataset/")
    train_test["test"].save_to_disk("tokenized_test_dataset/")
    
    print(f"Train set: {len(train_test['train'])} cells")
    print(f"Test set: {len(train_test['test'])} cells")
    
    return "tokenized_train_dataset/", "tokenized_test_dataset/"

# ============================================================================
# STEP 3: FINE-TUNE CLASSIFIER ON DISEASE STATES
# ============================================================================

def fine_tune_classifier(train_dataset_path, model_output_dir="disease_classifier"):
    """
    Fine-tune Geneformer classifier to learn disease states
    Training dataset: 7,000 cells total
    CORRECTED: Using proper training_args instead of non-existent parameters
    """
    print("=== STEP 3: FINE-TUNING CLASSIFIER ===")
    print(f"Training on 7,000 cells total")
    
    # Create output directories
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(f"{model_output_dir}/training_output", exist_ok=True)
    os.makedirs(f"{model_output_dir}/logs", exist_ok=True)
    os.makedirs(f"{model_output_dir}/prepared_data", exist_ok=True)
    
    # Define disease states to model
    cell_state_dict = {
        "state_key": "disease_group",
        "states": ["Active", "TNJDM", "Inactive", "HC"]
    }
    
    # Define training arguments (this is how you control epochs, learning rate, etc.)
    training_args = TrainingArguments(
        output_dir=f"{model_output_dir}/training_output",
        num_train_epochs=15,              # CORRECT: epochs go here
        learning_rate=1e-4,               # CORRECT: learning rate goes here
        per_device_train_batch_size=12,   # batch size
        per_device_eval_batch_size=12,
        warmup_steps=150,                 # warmup steps
        weight_decay=0.1,                 # regularization
        logging_dir=f"{model_output_dir}/logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    
    # Initialize classifier - CORRECTED parameters only
    classifier = Classifier(
        classifier="cell",
        cell_state_dict=cell_state_dict,
        filter_data=None,              # don't filter - use all your CD4 data
        max_ncells=None,               # use all 7,000 training cells
        max_ncells_per_class=None,     # no per-class limit
        training_args=training_args,   # CORRECT: pass training args here
        freeze_layers=6,               # freeze bottom 6 layers, fine-tune top 6
        num_crossval_splits=3,         # 3-fold CV for robust validation
        split_sizes={'train': 0.85, 'valid': 0.15, 'test': 0.0},  # use 85/15 split
        forward_batch_size=12,         # batch size for evaluation
        nproc=4,                       # number of processes
        ngpu=1                         # number of GPUs
    )
    
    # Prepare and train
    classifier.prepare_data(
        input_data_file=train_dataset_path,
        output_directory=f"{model_output_dir}/prepared_data",
        output_prefix="prepared"
    )
    
    # Run cross-validation with hyperparameter optimization
    all_metrics = classifier.validate(
        model_directory="../Geneformer/gf-12L-95M-i4096",  # pretrained model path
        prepared_input_data_file=f"{model_output_dir}/prepared_data/prepared_labeled.dataset",
        id_class_dict_file=f"{model_output_dir}/prepared_data/prepared_id_class_dict.pkl",
        output_directory=model_output_dir,
        output_prefix="disease_classifier",
        n_hyperopt_trials=10,          # CORRECT: hyperparameter optimization here
        save_eval_output=True,
        predict_eval=True
    )
    
    print(f"Fine-tuned classifier saved to {model_output_dir}")
    print(f"Cross-validation metrics: {all_metrics}")
    
    return model_output_dir

# ============================================================================
# STEP 4: RUN PERTURBATION ANALYSIS
# ============================================================================

def run_perturbation_analysis(test_dataset_path, classifier_path, output_dir="perturbation_results"):
    """
    Run in silico perturbation analysis to find therapeutic targets
    Test dataset: ~1,441 cells (20% of 7,201 total)
    """
    print("=== STEP 4: PERTURBATION ANALYSIS ===")
    print(f"Running perturbations on ~1,441 test cells")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define cell states to model for therapeutic discovery
    cell_states_to_model = {
        "state_key": "disease_group",
        "start_state": ["Active", "TNJDM"],      # diseased states
        "goal_state": "HC",                      # healthy control target
        "alt_states": ["Inactive"]               # other states
    }
    
    # Initialize perturber - using latest API
    isp = InSilicoPerturber(
        perturb_type="delete",          # try gene knockouts
        genes_to_perturb="all",         # test all genes
        combos=0,                       # single gene perturbations only
        model_type="CellClassifier",    # use your fine-tuned model
        num_classes=4,                  # number of disease states
        emb_mode="cls",                 # use CLS token embedding
        cell_emb_style="mean_pool",     # mean pooling for cell embeddings
        max_ncells=1500,                # use all test cells (~1,441)
        emb_layer=-1,                   # use final layer
        forward_batch_size=6,           # smaller batches for memory efficiency
        nproc=2,                        # conservative for stability
        cell_states_to_model=cell_states_to_model
    )
    
    # Run perturbation analysis
    isp.perturb_data(
        model_directory=classifier_path,
        input_data_file=test_dataset_path,
        output_directory=output_dir,
        output_prefix="therapeutic_targets"
    )
    
    print(f"Perturbation results saved to {output_dir}")
    return f"{output_dir}/therapeutic_targets_dict.pkl"

# ============================================================================
# STEP 5: CALCULATE PERTURBATION STATISTICS
# ============================================================================

def calculate_perturbation_stats(perturbation_dict_path, output_dir="perturbation_stats"):
    """
    Calculate statistics from perturbation results
    """
    print("=== STEP 5: CALCULATING STATISTICS ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize stats calculator
    ispstats = InSilicoPerturberStats(
        mode="goal_state_shift",          # focus on therapeutic state shifts
        genes_perturbed="all",
        combos=0,
        anchor_gene=None,
        cell_states_to_model={
            "state_key": "disease_group",
            "start_state": ["Active", "TNJDM"],
            "goal_state": "HC",
            "alt_states": ["Inactive"]
        }
    )
    
    # Calculate stats
    ispstats.get_stats(
        input_data_directory=os.path.dirname(perturbation_dict_path),
        null_dist_data_directory=None,  # optional: for statistical significance
        output_directory=output_dir,
        output_prefix="therapeutic_stats"
    )
    
    print(f"Statistics saved to {output_dir}")
    return f"{output_dir}/therapeutic_stats.csv"

# ============================================================================
# STEP 6: ANALYZE RESULTS
# ============================================================================

def analyze_results(stats_file):
    """
    Analyze and interpret perturbation statistics
    """
    print("=== STEP 6: ANALYZING RESULTS ===")
    
    # Load results
    results = pd.read_csv(stats_file)
    
    # Sort by therapeutic potential (goal state shift)
    results_sorted = results.sort_values('Goal_state_shift', ascending=False)
    
    print("\n=== TOP THERAPEUTIC TARGETS ===")
    print("Genes whose deletion best shifts diseased cells toward healthy state:")
    print(results_sorted[['Gene_name', 'Goal_state_shift', 'Alt_state_shift']].head(10))
    
    print("\n=== INTERPRETATION ===")
    print("- Higher Goal_state_shift = better therapeutic potential")
    print("- These genes, when targeted, could reverse disease phenotype")
    print("- Prioritize for experimental validation")
    
    return results_sorted

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Run complete pipeline from raw data to therapeutic targets
    """
    print("STARTING COMPLETE GENEFORMER PERTURBATION PIPELINE")
    print("Using Latest Geneformer API v0.1.0 (2024)")
    print("=" * 60)
    
    # Configuration
    raw_data_path = "your_data.h5ad"  # UPDATE THIS PATH
    
    try:
        # Step 1: Tokenize data
        tokenized_path = tokenize_data(raw_data_path)
        
        # Step 2: Split into train/test
        train_path, test_path = split_dataset(tokenized_path)
        
        # Step 3: Fine-tune classifier
        classifier_path = fine_tune_classifier(train_path)
        
        # Step 4: Run perturbation analysis
        perturbation_results = run_perturbation_analysis(test_path, classifier_path)
        
        # Step 5: Calculate statistics
        stats_file = calculate_perturbation_stats(perturbation_results)
        
        # Step 6: Analyze results
        final_results = analyze_results(stats_file)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Final results saved to: {stats_file}")
        print("=" * 60)
        
        return final_results
        
    except Exception as e:
        print(f"ERROR in pipeline: {str(e)}")
        raise

# ============================================================================
# QUICK TEST VERSION (Updated with correct API)
# ============================================================================

def quick_test():
    """
    Quick test version using subset of data with correct API
    """
    print("=== RUNNING QUICK TEST WITH CORRECT API ===")
    
    # Load and subset data
    full_dataset = load_from_disk("tokenized_train_dataset/tokenized.dataset")
    subset = full_dataset.select(range(100))  # slightly more for meaningful test
    subset_path = "tokenized_train_dataset/subset_test.dataset"
    subset.save_to_disk(subset_path)

    os.makedirs("quick_test_output", exist_ok=True)

    
    # Define minimal training args for quick test
    quick_training_args = TrainingArguments(
        output_dir="quick_test_output",
        num_train_epochs=3,              # just 3 epochs for quick test
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="no",        # no eval for quick test
        save_strategy="no",              # don't save for quick test
        remove_unused_columns=False,
    )
    
    # Quick classifier test
    quick_classifier = Classifier(
        classifier="cell",
        cell_state_dict={"state_key": "disease_group", "states": "all"},
        max_ncells=100,                  # use only 100 cells
        training_args=quick_training_args,
        freeze_layers=8,                 # freeze more layers for speed
        num_crossval_splits=1,           # no cross-validation
        forward_batch_size=8,
        nproc=2,
        ngpu=1
    )

    os.makedirs("quick_test_prepared", exist_ok=True)
    os.makedirs("quick_test_results", exist_ok=True)

    # Quick preparation and validation
    quick_classifier.prepare_data(
        input_data_file=subset_path,
        output_directory="quick_test_prepared",
        output_prefix="quick"
    )
    
    quick_metrics = quick_classifier.validate(
        model_directory="../Geneformer/gf-12L-95M-i4096",
        prepared_input_data_file="quick_test_prepared/quick_labeled.dataset",
        id_class_dict_file="quick_test_prepared/quick_id_class_dict.pkl",
        output_directory="quick_test_results",
        output_prefix="quick_classifier",
        n_hyperopt_trials=0,             # no hyperopt for quick test
        save_eval_output=False
    )
    
    print("Quick test completed!")
    print(f"Quick test metrics: {quick_metrics}")

if __name__ == "__main__":
    # For quick testing with correct API
    quick_test()
    
    # For full pipeline
    # main()
