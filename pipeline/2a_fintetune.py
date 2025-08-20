import os
import random
import math
import pandas as pd
from tqdm import tqdm
from transformers import TrainingArguments
from geneformer import Classifier

# Paths
input_data_folder = os.path.abspath("../tokenized_dataset/CD4All_JDM.dataset")
prepared_data_folder = os.path.abspath("../CD4_finetune_prepared")
results_folder = os.path.abspath("../CD4_finetune_results_aug19_25trials")
model_path = os.path.abspath("../Geneformer/Geneformer-V2-104M")

os.makedirs(prepared_data_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

# Donor splits
train_ids = ['HC16', 'HC17', 'HC18', 'JDM2', 'JDM3', 'JDM4', 'JDM5', 'JDM6', 'JDM7']
eval_ids = ['HC19', 'JDM8']
test_ids = ['HC20', 'JDM9', 'JDM10']

train_test_id_split_dict = {
    "attr_key": "donor_id",
    "train": train_ids + eval_ids,
    "test": test_ids
}

train_valid_id_split_dict = {
    "attr_key": "donor_id",
    "train": train_ids,
    "eval": eval_ids
}

# Prepare data once
model = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "disease_group", "states": ["HC", "TNJDM"]},
    training_args={},  # placeholder
    freeze_layers=4,
    num_crossval_splits=1,
    forward_batch_size=64,
    nproc=12,
    model_version="V2"
)

model.prepare_data(
    input_data_file=input_data_folder,
    output_directory=prepared_data_folder,
    output_prefix="CD4ALL",
    split_id_dict=train_test_id_split_dict
)

# Random sampling helpers
def sample_log_uniform(low, high):
    """Sample from log-uniform distribution."""
    return math.exp(random.uniform(math.log(low), math.log(high)))

def sample_uniform(low, high):
    return random.uniform(low, high)

def sample_params():
    return {
        "learning_rate": sample_log_uniform(1e-5, 5e-3),
        "weight_decay": sample_log_uniform(1e-4, 0.2),
        "warmup_steps": random.randint(100, 1000),
        "batch_size": random.choice([4, 8]),
        "seed": random.randint(1, 9999),
        "gradient_accumulation_steps": random.choice([4, 8, 16])
    }

# Run settings
num_trials = 25
results_filename = os.path.join(results_folder, "hyperparam_search_results_aug19_25trials.csv")

# Create a list to track in-memory results for the leaderboard
live_results_list = []

print("Begin probabilistic hyperparameter search")
pbar = tqdm(range(num_trials), desc="Trials", ncols=100)

for i in pbar:
    params = sample_params()
    run_prefix = f"run{i + 1}"

    pbar.set_postfix({
        "lr": f"{params['learning_rate']:.2e}",
        "wd": f"{params['weight_decay']:.2e}",
        "bs": params["batch_size"],
        "acc_steps": params["gradient_accumulation_steps"],
        "seed": params["seed"]
    })

    training_args = TrainingArguments(
        num_train_epochs=1,
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=params["batch_size"],
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
        lr_scheduler_type='linear',
        warmup_steps=params["warmup_steps"],
        weight_decay=params["weight_decay"],
        seed=params["seed"],
        report_to="wandb",
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        fp16=True
    )

    model.training_args = training_args.to_dict()

    metrics = model.validate(
        model_directory=model_path,
        prepared_input_data_file=f"{prepared_data_folder}/CD4ALL_labeled_train.dataset",
        id_class_dict_file=f"{prepared_data_folder}/CD4ALL_id_class_dict.pkl",
        output_directory=results_folder,
        output_prefix=f"{run_prefix}_jdm_classifier_CD4ALL",
        split_id_dict=train_valid_id_split_dict,
        n_hyperopt_trials=0
    )

    # Record results and hyperparameters
    trial_result = {
        "run": run_prefix,
        "learning_rate": params["learning_rate"],
        "weight_decay": params["weight_decay"],
        "warmup_steps": params["warmup_steps"],
        "batch_size": params["batch_size"],
        "gradient_accumulation_steps": params["gradient_accumulation_steps"],
        "seed": params["seed"],
        "acc": float(metrics["acc"][0]) if isinstance(metrics.get("acc"), list) else float(metrics["acc"]),
        "macro_f1": float(metrics["macro_f1"][0]) if isinstance(metrics.get("macro_f1"), list) else float(metrics["macro_f1"]),
        "roc_auc": float(metrics["all_roc_metrics"]["roc_auc"]) if "all_roc_metrics" in metrics else float("nan")
    }
    
    # Append the results to the in-memory list for the live leaderboard
    live_results_list.append(trial_result)

    # Convert the current trial's results to a DataFrame
    trial_df = pd.DataFrame([trial_result])

    # Check if the file already exists to decide whether to write the header
    write_header = not os.path.exists(results_filename)
    trial_df.to_csv(results_filename, mode='a', header=write_header, index=False)
    
    # Update live leaderboard
    leaderboard = sorted(live_results_list, key=lambda x: x["macro_f1"], reverse=True)[:5]
    leaderboard_df = pd.DataFrame(leaderboard)
    pbar.write("\n--- Leaderboard (Top 5 by macro_f1) ---")
    pbar.write(leaderboard_df.to_string(index=False))

print("\nAll runs complete! Results saved to hyperparam_search_results.csv")