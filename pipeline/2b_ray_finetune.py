import os
import random
import math
import pandas as pd
from tqdm import tqdm
from transformers import TrainingArguments
from geneformer import Classifier

# Ray imports
import ray
from ray import train, tune # ⭐ New: Import ray.train
from ray.air import session 
from ray.tune.search.hyperopt import HyperOptSearch

# WandB import
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false" 

# Paths
input_data_folder = os.path.abspath("../tokenized_dataset/CD4All_JDM.dataset")
prepared_data_folder = os.path.abspath("../CD4_finetune_prepared")
results_folder = os.path.abspath("../CD4_finetune_results_ray_run")
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

# Ray Tune Trainable Function
def hyperopt_trainable(config):
    wandb.init(
        project="geneformer-hyperopt-cd4",
        config=config,
        name=f"trial-{session.get_trial_id()}-{config['learning_rate']:.2e}"
    )

    trial_id = session.get_trial_id()
    trial_out_dir = os.path.join(results_folder, f"trial_{trial_id}")
    os.makedirs(trial_out_dir, exist_ok=True)
    trial_prefix = f"ray_trial_{trial_id}"

    model = Classifier(
        classifier="cell",
        cell_state_dict={"state_key": "disease_group", "states": ["HC", "TNJDM"]},
        training_args={},
        freeze_layers=4,
        num_crossval_splits=1,
        forward_batch_size=64,
        nproc=12,
        model_version="V2"
    )

    training_args = TrainingArguments(
        num_train_epochs=1,
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        lr_scheduler_type='linear',
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        seed=config["seed"],
        report_to="wandb",
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        fp16=True,
        output_dir=trial_out_dir,
    )
    model.training_args = training_args.to_dict()

    metrics = model.validate(
        model_directory=model_path,
        prepared_input_data_file=f"{prepared_data_folder}/CD4ALL_labeled_train.dataset",
        id_class_dict_file=f"{prepared_data_folder}/CD4ALL_id_class_dict.pkl",
        output_directory=trial_out_dir,
        output_prefix=trial_prefix,
        split_id_dict=train_valid_id_split_dict,
        n_hyperopt_trials=0
    )

    macro_f1 = metrics["macro_f1"][0] if isinstance(metrics.get("macro_f1"), list) else metrics.get("macro_f1")
    acc = metrics["acc"][0] if isinstance(metrics.get("acc"), list) else metrics.get("acc")
    roc_auc = float("nan") if metrics.get("all_roc_metrics") is None or "roc_auc" not in metrics["all_roc_metrics"] else metrics["all_roc_metrics"]["roc_auc"]

    # ⭐ Corrected: Create a dictionary of metrics to report
    report_metrics = {
        "macro_f1": float(macro_f1) if macro_f1 is not None else float("nan"),
        "acc": float(acc) if acc is not None else float("nan"),
        "roc_auc": float(roc_auc)
    }

    # ⭐ Pass the dictionary to train.report()
    train.report(report_metrics)
    wandb.log(report_metrics)
    wandb.finish()


# Prepare data once
model = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "disease_group", "states": ["HC", "TNJDM"]},
    training_args={},
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

ray.init()

search_space = {
    "learning_rate": tune.loguniform(1e-5, 5e-3),
    "weight_decay": tune.loguniform(1e-4, 0.2),
    "warmup_steps": tune.randint(100, 1000),
    "batch_size": tune.choice([4, 8]),
    "gradient_accumulation_steps": tune.choice([4, 8, 16]),
    "seed": tune.randint(1, 9999)
}

hyperopt_search = HyperOptSearch(metric="macro_f1", mode="max")

print("Begin Ray Tune hyperparameter search")

trainable = tune.with_resources(hyperopt_trainable, {"cpu": 6, "gpu": 1})
tuner = tune.Tuner(
    trainable,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=25,
        search_alg=hyperopt_search,
    ),
    run_config=ray.air.RunConfig(
        name="jdm_classifier_hyperopt_ray",
        storage_path=results_folder,
    )
)

results = tuner.fit()

best_result = results.get_best_result(metric="macro_f1", mode="max")
print("\nBest trial configuration:", best_result.config)
print("Best trial metrics:", best_result.metrics)

df_results = results.get_dataframe()
df_csv = os.path.join(results_folder, "ray_hyperopt_results.csv")
df_results.to_csv(df_csv, index=False)
print("\nAll Ray Tune results saved to ray_hyperopt_results.csv")