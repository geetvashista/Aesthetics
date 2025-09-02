import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from scipy.stats import false_discovery_control
from statsmodels.stats.multitest import fdrcorrection

# H1 vs. L1
# Change directory: Art Naive (alpha, gt), (beta, gt), (theta, gt) / Art Sensitive (alpha, gt), (beta, gt), (theta, gt)

# Set up directory
base_path = r"C:\Users\tmcc380\OneDrive - The University of Auckland\Desktop\theta_graph_metrics\art_sensitive\strength" # Art Sensitive

# Output path
output_path = r"C:\Users\tmcc380\OneDrive - The University of Auckland\Desktop\theta_graph_metrics\art_sensitive"

# Output folder
output_folder = os.path.join(output_path, "all_results")
os.makedirs(output_folder, exist_ok=True)
sig_output_folder = os.path.join(output_path, "sig_results")
os.makedirs(sig_output_folder, exist_ok=True)

# Get all files
all_h1_files = [f for f in glob.glob(os.path.join(base_path, "*npz")) if '_h1_' in os.path.basename(f)]
all_l1_files = [f for f in glob.glob(os.path.join(base_path, "*npz")) if '_l1_' in os.path.basename(f)]

# Load first file to get ROIs
first_file = all_h1_files[0]
npz = np.load(first_file)
labels = npz['labels']
roi_labels = [label.split("'")[1] for label in labels]

# Load all h1 files
all_h1_data = []
for file in all_h1_files:
    h1_npz = np.load(file, allow_pickle=True)
    h1_data = h1_npz['strength']
    all_h1_data.append(h1_data)

# Load all l1 files
all_l1_data = []
for file in all_l1_files:
    l1_npz = np.load(file, allow_pickle=True)
    l1_data = l1_npz['strength']
    all_l1_data.append(l1_data)

h1_array = np.array(all_h1_data)    # (n_participants, n_rois)
l1_array = np.array(all_l1_data)    # (n_participants, n_rois)
diff_array = np.subtract(h1_array, l1_array)    # (n_participants, n_rois)

# Perform ROI-wise paired t-tests
t_vals, p_vals = ttest_rel(h1_array, l1_array, axis=0)

# False discovery control
p_vals = np.nan_to_num(p_vals, nan=1.0)
fdc_p_vals = false_discovery_control(p_vals)

# Mean difference to determine direction
mean_diffs = np.mean(diff_array, axis=0)
all_results_df = pd.DataFrame({"ROI": roi_labels,
                               "t-values": t_vals,
                               "p-values": p_vals,
                               "FDC p-values": fdc_p_vals,
                               "Mean Difference": mean_diffs,
                               "Direction": np.where(mean_diffs > 0, "Increase", "Decrease")}
                              )
all_results_df.to_csv(os.path.join(output_folder, "as_theta_strength_results.csv"), index=False)

# Calculate and save significant results
sig_results = []

for i, p in enumerate(p_vals):
    if p < 0.05:
        roi = roi_labels[i]
        sig_results.append({"ROI": roi,
                            "t-values": t_vals[i],
                            "p-values": p,
                            "FDC p-values": fdc_p_vals[i],
                            "Mean Difference": mean_diffs[i],
                            "Direction": "Increase" if mean_diffs[i] > 0 else "Decrease"}
                           )

if sig_results:
    sig_results_df = pd.DataFrame(sig_results)
    sig_results_df.to_csv(
        os.path.join(sig_output_folder, "0.05_sig_as_theta_strength_results.csv"),
        index=False)
else:
    print("No significant components found")

