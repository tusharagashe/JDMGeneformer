import os
import torch
from transformers import TrainingArguments
from geneformer import Classifier
from ray import tune


input_data_folder = os.path.abspath("tokenized_dataset/tokenized.dataset")
prepared_data_folder = os.path.abspath("CD4_finetune_prepared")
results_folder = os.path.abspath("CD4_finetune_results")
model_path = os.path.abspath("../Geneformer/Geneformer-V2-104M")


training_args = TrainingArguments(
      num_train_epochs=1,              
      learning_rate=0.0009666412211122036,
      per_device_train_batch_size=16,
      lr_scheduler_type='linear',
      warmup_steps=700,
      weight_decay=0.1312456571918533,
      seed=38,
  )




model = Classifier(
      classifier="cell",
      cell_state_dict={"state_key": "disease_group", "states": ["HC","TNJDM"]},
      training_args=training_args.to_dict(),
      freeze_layers=4,                 
      num_crossval_splits=1,           # no cross-validation
      forward_batch_size=64,
      nproc=30,
      model_version="V2")

os.makedirs(prepared_data_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

train_ids = ['HC16', 'HC17', 'HC18', 'JDM2', 'JDM3', 'JDM4', 'JDM5', 'JDM7', 'JDM8', 'JDM9']
eval_ids = ['HC19', 'JDM6']
test_ids = ['HC20', 'JDM10']

train_test_id_split_dict = {"attr_key": "donor_id",
                            "train": train_ids+eval_ids,
                            "test": test_ids}


model.prepare_data(
      input_data_file=input_data_folder,
      output_directory=prepared_data_folder,
      output_prefix="jdm",
      split_id_dict=train_test_id_split_dict
)


train_valid_id_split_dict = {"attr_key": "donor_id",
                            "train": train_ids,
                            "eval": eval_ids}

print("Begin classifier training")

metrics = model.validate(
        model_directory=model_path,
        prepared_input_data_file=f"{prepared_data_folder}/jdm_labeled_train.dataset",
        id_class_dict_file=f"{prepared_data_folder}/jdm_id_class_dict.pkl",
        output_directory=results_folder,
        output_prefix="jdm_classifier_with_no_hyperopt",
        split_id_dict=train_valid_id_split_dict,
        n_hyperopt_trials=0
    )    

print("Classifier training complete!")
print(f"JDM Classifier metrics: {metrics}")