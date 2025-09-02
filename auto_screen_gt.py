import bct
import numpy as np
import os
import glob

# Set up wpli base path
wpli_base_path = r"N:\screen\processed_files\auto\processed_data_fif\epochs\epochs_for_analysis\stcs\theta_stcs\parcellation\wpli"

# Get all files
all_npz_files = glob.glob(os.path.join(wpli_base_path, '*.npz'))
all_base_names = [os.path.splitext(os.path.basename(f))[0].replace('-wpli', '') for f in all_npz_files]

# Set up output folders
graph_output_folder = os.path.join(wpli_base_path, 'graph_metrics')
os.makedirs(graph_output_folder, exist_ok=True)

# Set output dirs
strength_dir = os.path.join(graph_output_folder, 'strength')
os.makedirs(strength_dir, exist_ok=True)

betweenness_dir = os.path.join(graph_output_folder, 'betweenness')
os.makedirs(betweenness_dir, exist_ok=True)

eigenvector_dir = os.path.join(graph_output_folder, 'eigenvector')
os.makedirs(eigenvector_dir, exist_ok=True)

clustering_dir = os.path.join(graph_output_folder, 'clustering')
os.makedirs(clustering_dir, exist_ok=True)

def graph_metrics(all_npz_files, all_base_names):

    strength_all = []
    betweenness_all = []
    eigenvector_all = []
    clustering_all = []

    for file, base_name in zip(all_npz_files, all_base_names):
        npz = np.load(file, allow_pickle=True)
        adj_matrix = npz['wpli']
        labels = npz['labels']

        strength = bct.strengths_und(np.nan_to_num(adj_matrix))
        strength_all.append(strength)

        np.savez(
            os.path.join(strength_dir, f'{base_name}-strength.npz'),
            strength=strength,
            labels=labels,
            overwrite=True
        )

        betweenness = bct.betweenness_wei(np.nan_to_num(adj_matrix))
        betweenness_all.append(betweenness)

        np.savez(
            os.path.join(betweenness_dir, f'{base_name}-betweenness.npz'),
            betweenness=betweenness,
            labels=labels,
            overwrite=True
        )

        eigenvector = bct.eigenvector_centrality_und(np.nan_to_num(adj_matrix))
        eigenvector_all.append(eigenvector)

        np.savez(
            os.path.join(eigenvector_dir, f'{base_name}-eigenvector.npz'),
            eigenvector=eigenvector,
            labels=labels,
            overwrite=True
        )

        clustering = bct.clustering_coef_wu(np.nan_to_num(adj_matrix))
        clustering_all.append(clustering)

        np.savez(
            os.path.join(clustering_dir, f'{base_name}-clustering.npz'),
            clustering=clustering,
            labels=labels,
            overwrite=True
        )

graph_metrics(all_npz_files, all_base_names)
