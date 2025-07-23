

from ray.tune import ExperimentAnalysis
import os 
import pandas

ray_results = os.path.abspath("CD4_finetune_results/250723_geneformer_cellClassifier_jdm_classifier_with_hyperopt_after_split_update/ksplit1/_objective_2025-07-23_02-49-31")

analysis = ExperimentAnalysis(ray_results)

df = analysis.dataframe()
print(df.sort_values(by="eval_macro_f1", ascending=False).head())
df.to_csv("binaryclass_post_split_update.csv")