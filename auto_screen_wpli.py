import dyconnmap
import bct
import numpy as np
import os
import time
import glob

# === wPLI ===
# Set up parcellation base path
parcellation_base_path = r"N:\screen\processed_files\auto\processed_data_fif\epochs\epochs_for_analysis\stcs\beta_stcs\parcellation"

# Get all files
all_npz_files = glob.glob(os.path.join(parcellation_base_path, '*.npz'))
all_base_names = [os.path.splitext(os.path.basename(f))[0].replace('-prcl', '') for f in all_npz_files]

# Define target_fb and fs
target_fb = [13, 30]  # Theta [4, 8], Alpha [8, 13], Beta [13, 30]
fs = 250

# Set up output folders
wpli_output_folder = os.path.join(parcellation_base_path, 'wpli')
os.makedirs(wpli_output_folder, exist_ok=True)

start = time.time()

def wpli_conn(all_npz_files, all_base_names, target_fb, fs, wpli_output_folder):
    all_matrices = []

    for file_path, base_name in zip(all_npz_files, all_base_names):
        npz = np.load(file_path, allow_pickle=True)
        data_array = npz['data_array']
        labels = npz['labels']

        # Compute wpli
        adj_matrix = dyconnmap.fc.wpli(data_array, fb=target_fb, fs=fs)

        # Zero out negative values
        adj_matrix[adj_matrix < 0] = 0

        # Remove self-loops (matrix diagonal)
        np.fill_diagonal(adj_matrix, 0)

        # Append all matrices
        all_matrices.append({
            'participant': base_name,
            'wpli': adj_matrix,
            'labels': labels})

        np.savez(os.path.join(wpli_output_folder, f'{base_name}-wpli.npz'), wpli=adj_matrix, labels=labels, overwrite=True)

    print(f'Saved {len(all_npz_files)} wPLI matrices to {wpli_output_folder}')

    return all_matrices

wpli = wpli_conn(all_npz_files, all_base_names, target_fb, fs, wpli_output_folder)

end = time.time()
print("Execution time: ", end - start, "seconds")
