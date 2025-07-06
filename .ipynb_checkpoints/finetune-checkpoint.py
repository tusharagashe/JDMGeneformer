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
      learning_rate=5e-5,
      per_device_train_batch_size=12,
      lr_scheduler_type='polynomial',
      warmup_steps=50,
      weight_decay=0.01,
      seed=73,
  )



# ray_config = {
#     "learning_rate": tune.loguniform(2e-4, 8e-4), 
#     "weight_decay": tune.uniform(0.02, 0.06),     
#     "warmup_steps": tune.randint(400, 700),        
#     "lr_scheduler_type": tune.choice(["polynomial", "cosine", "linear"]),
#     "num_train_epochs": tune.choice([1]),          
#     "per_device_train_batch_size": tune.choice([12]),  
#     "seed": tune.randint(0, 100)                   
# }


model = Classifier(
      classifier="cell",
      cell_state_dict={"state_key": "disease_group", "states": "all"},
      max_ncells=None,                  
      training_args=training_args.to_dict(),
      freeze_layers=4,                 
      num_crossval_splits=1,           # no cross-validation
      forward_batch_size=64,
      nproc=12,
      model_version="V2")

os.makedirs(prepared_data_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

train_ids = [5, 9, 14, 17, 4, 10, 1, 12, 0, 15, 19, 16, 12, 14, 5, 9] # all cases ( 3 HCs, 6 TNJDM, 3 Active, 4 Inactive)
eval_ids = [6, 7, 3, 11] # 1 case each
test_ids = [2, 8, 13, 18] # 1 case each 

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
        output_prefix="jdm_classifier_with_hyperopt",
        split_id_dict=train_valid_id_split_dict,
        n_hyperopt_trials=120
    )    

print("Classifier training complete!")
print(f"JDM Classifier metrics: {metrics}")