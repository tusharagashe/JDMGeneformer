import os
import pandas as pd
import scanpy as sc
from datasets import load_from_disk
from geneformer import Classifier



cell_state_dict = {
    "state_key": "disease_group",
    "states": ["Active", "TNJDM", "Inactive", "HC"]
}

classifier = Classifier(
    classifier="cell",
    cell_state_dict=cell_state_dict,
    filter_data=None,  
    max_ncells=None,   
    max_epochs=15,     
    learning_rate=1e-4, 
    freeze_layers=6,   # freeze bottom 6 layers, fine-tune top 6
    num_crossval_splits=3,  # 3-fold CV for robust validation
    eval_size=0.15,   
    n_hyperopt_trials=3,  
    output_dir="finetunedmodel",
    output_prefix="disease_classifier",
    batch_size=12,     # smaller batch size for stability
    warmup_steps=150,  # warmup for stable training (increased for larger dataset)
    weight_decay=0.1   # regularization to prevent overfitting
)

classifier.prepare_data(
    input_data_file="tokenized_train_dataset/tokenized.dataset",
    output_directory="finetunedmodel/prepared_data",
    output_prefix="prepared"
)

classifier.validate(
    model_directory="../Geneformer/gf-12L-95M-i4096",  # pretrained model path
    prepared_input_data_file="finetunedmodel/prepared_data/prepared.dataset",
    id_class_dict_file="finetunedmodel/prepared_data/id_class_dict.pkl",
    output_directory="finetunedmodel",
    output_prefix="disease_classifier"
)

print(f"Fine-tuned classifier saved to finetunedmodel/")
