import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----- User-defined variables -----
eval_predictions_file = os.path.abspath("CD4_finetune_results_aug19_50trials/model_eval_split_update_aug19_50trials_jdm_classifier_CD4ALL_pred_dict.pkl")

# Load the saved predictions
with open(eval_predictions_file, 'rb') as f:
    prediction_dict = pickle.load(f)

# Extract true labels and TNJDM probabilities
true_labels = np.array(prediction_dict["label_ids"])
tnjdm_probs = np.array([p[1] for p in prediction_dict["predictions"]])

# Separate probabilities by true class
tnjdm_probs_tnjdm = tnjdm_probs[true_labels == 1]
tnjdm_probs_hc = tnjdm_probs[true_labels == 0]

# Plot the distributions
plt.figure(figsize=(10, 6))
sns.kdeplot(tnjdm_probs_hc, label='True HC', color='blue', shade=True)
sns.kdeplot(tnjdm_probs_tnjdm, label='True TNJDM', color='red', shade=True)
plt.axvline(x=0.01, color='green', linestyle='--', label='Optimal Threshold (0.01)')
plt.title('Distribution of TNJDM Probability Scores')
plt.xlabel('Predicted Probability of being TNJDM')
plt.ylabel('Density')
plt.legend()


# Save the plot instead of showing it
plot_file = os.path.join("CD4_finetune_results_aug19_50trials", "probability_distribution.png")
plt.savefig(plot_file)
print(f"Plot saved to: {plot_file}")
plt.close() # Close the plot to free up memory