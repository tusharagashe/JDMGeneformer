import os
import pickle
import numpy as np
from scipy.special import expit # This is the sigmoid function
from sklearn.metrics import f1_score


# Path to your evaluation set's prediction dictionary
eval_predictions_file = os.path.abspath("CD4_finetune_results_ray_run/aug19_ray_run_jdm_classifier_CD4ALL_pred_dict.pkl")

# Load the saved predictions
with open(eval_predictions_file, 'rb') as f:
    prediction_dict = pickle.load(f)

# Extract true labels and raw logit scores
true_labels = np.array(prediction_dict["label_ids"])
logits = np.array([p[1] for p in prediction_dict["predictions"]])

# ----- New Step: Transform Logits to Probabilities -----
# Apply the sigmoid function to convert logits to a 0-1 probability scale
tnjdm_probs = expit(logits)

# Find the optimal threshold
best_f1 = 0
best_threshold = 0.5
thresholds = np.arange(0.01, 1.0, 0.01) # Search on the probability scale

for threshold in thresholds:
    predicted_labels = (tnjdm_probs > threshold).astype(int)
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Optimal Threshold on the Probability Scale: {best_threshold:.2f} with Macro F1: {best_f1:.2f}")

