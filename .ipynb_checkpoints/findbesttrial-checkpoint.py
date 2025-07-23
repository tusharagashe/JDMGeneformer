

from ray.tune import ExperimentAnalysis
import os 
import pandas

ray_results = os.path.abspath("CD4_finetune_results/250722_geneformer_cellClassifier_jdm_classifier_with_no_hyperopt/ksplit1/_objective_2025-07-22_17-33-02")

analysis = ExperimentAnalysis(ray_results)

df = analysis.dataframe()
print(df.sort_values(by="eval_macro_f1", ascending=False).head())
df.to_csv("binaryclass_latest_hyperopt.csv")