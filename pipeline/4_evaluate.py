import os
from geneformer import Classifier


input_data_folder = os.path.abspath("tokenized_dataset/tokenized.dataset")
prepared_data_folder = os.path.abspath("CD4_finetune_prepared")
results_folder = os.path.abspath("CD4_finetune_results")
model_path = os.path.abspath("savedmodels/run-ce0ee068/checkpoint-111")



cc = Classifier(classifier="cell",
                cell_state_dict = {"state_key": "disease_group", "states": ["HC", "TNJDM"]},
                forward_batch_size=64,
                nproc=24)



all_metrics_test = cc.evaluate_saved_model(
        model_directory=model_path,
        id_class_dict_file=f"{prepared_data_folder}/jdm_id_class_dict.pkl",
        test_data_file=f"{prepared_data_folder}/jdm_labeled_test.dataset",
        output_directory=results_folder,
        output_prefix="model_eval_split_update_aug2",
    )

print(all_metrics_test)

cc.plot_conf_mat(
        conf_mat_dict={"Geneformer": all_metrics_test["conf_matrix"]},
        output_directory=results_folder,
        output_prefix="model_eval_split_update_aug2",
)