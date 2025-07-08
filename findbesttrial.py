

from ray.tune import ExperimentAnalysis
import os 
import pandas

ray_results = os.path.abspath("CD4_finetune_results/250707_geneformer_cellClassifier_jdm_classifier_with_hyperopt/ksplit1/_objective_2025-07-07_22-52-14/")

analysis = ExperimentAnalysis(ray_results)

df = analysis.dataframe()
print(df.sort_values(by="eval_macro_f1", ascending=False).head())
df.to_csv("alltrials_1000.csv")