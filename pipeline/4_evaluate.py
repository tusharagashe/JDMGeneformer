import os
from geneformer import Classifier


input_data_folder = os.path.abspath("../tokenized_dataset/tokenized.dataset")
prepared_data_folder = os.path.abspath("../CD4_finetune_prepared")
results_folder = os.path.abspath("../CD4_finetune_results_ray_run")
model_path = os.path.abspath("../CD4_finetune_results_ray_run/trial_1da7fc18/250820_geneformer_cellClassifier_ray_trial_1da7fc18/ksplit1/checkpoint-125")



cc = Classifier(classifier="cell",
                cell_state_dict = {"state_key": "disease_group", "states": ["HC", "TNJDM"]},
                forward_batch_size=16,
                nproc=8)



all_metrics_test = cc.evaluate_saved_model(
        model_directory=model_path,
        id_class_dict_file=f"{prepared_data_folder}/CD4ALL_id_class_dict.pkl",
        test_data_file=f"{prepared_data_folder}/CD4ALL_labeled_test.dataset",
        output_directory=results_folder,
        output_prefix="aug19_ray_run_jdm_classifier_CD4ALL",
    )

print(all_metrics_test)

cc.plot_conf_mat(
        conf_mat_dict={"Geneformer": all_metrics_test["conf_matrix"]},
        output_directory=results_folder,
        output_prefix="aug19_ray_run_jdm_classifier_CD4ALL",
)