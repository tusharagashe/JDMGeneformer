import os
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit

# ----- User-defined variables -----
# Path to your test set's prediction dictionary
test_predictions_file = os.path.abspath("CD4_finetune_results_ray_run/aug19_ray_run_jdm_classifier_CD4ALL_pred_dict.pkl")
# The optimal threshold found from your calibration
optimal_threshold = 0.16 
# Directory to save the output plot
results_folder = os.path.abspath("CD4_finetune_results_ray_run")
plot_file = os.path.join(results_folder, "corrected_final_conf_matrix.png")

# Load the saved predictions for the test set
with open(test_predictions_file, 'rb') as f:
    prediction_dict = pickle.load(f)

# The correct order of labels for the confusion matrix
labels = ["HC", "TNJDM"]

# Extract true labels and raw logit scores
# CORRECTED: Use the integer labels directly
true_labels = np.array(prediction_dict["label_ids"])
logits = np.array([p[1] for p in prediction_dict["predictions"]])

# Apply the sigmoid function to convert logits to a 0-1 probability scale
tnjdm_probs = expit(logits)

# Apply the optimal threshold to get the final predictions
# Note: The predictions will be 0 for HC and 1 for TNJDM, which is what we need.
final_predictions = (tnjdm_probs > optimal_threshold).astype(int)

# Generate the confusion matrix using the correct label order
conf_matrix = confusion_matrix(true_labels, final_predictions, labels=[0, 1])

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Optimized Confusion Matrix (Threshold: {optimal_threshold:.2f})')

# Save the plot
plt.savefig(plot_file)
print(f"Corrected confusion matrix saved to: {plot_file}")
plt.close()