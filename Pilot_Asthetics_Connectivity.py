# This is script to calculate connectivity using raw np arrays,
# each assumed to be in the shape (channels/ROI's, time points)

import numpy as np
import time
import dyconnmap
import bct
start = time.time()

output_dir = '/home/sahib/Documents/Tay_Pilot/'
data = np.load('/home/sahib/Downloads/10s_p01_stc_parcellated_array.npy')
array_type = 'beta_'
target_fb = [8, 13]
fs = 250

def wpli_conn(array, target_fb, fs):   # (participants, roi's, time_points)
    adj_matrix = []
    for participant in array:
        adj_matrix.append(dyconnmap.fc.wpli(participant, fs=fs, fb=target_fb))
    return np.array(adj_matrix)


def graph_metrics(adj_matrix):  # TODO: Add in stats, maybe nbs? Could use fdc as well?
    # Strength calculator
    Strength = []
    for participant in adj_matrix:
        Strength.append(bct.strengths_und(np.nan_to_num(participant)))
    Strength = np.array(Strength)

    # Zeroing negative phasing
    Strength[Strength < 0] = 0

    # Betweenness centrality calculator
    Betweenness = []
    for participant in adj_matrix:
        Betweenness.append(bct.betweenness_wei(np.nan_to_num(participant)))
    Betweenness = np.array(Betweenness)

    # Eigenvector centrality calculator
    Eigenvector = []
    for participant in adj_matrix:
        Eigenvector.append(bct.eigenvector_centrality_und(np.nan_to_num(participant)))
    Eigenvector = np.array(Eigenvector)

    # Clustering calculator
    Clustering = []
    for participant in adj_matrix:
        Clustering.append(bct.clustering_coef_wu(np.nan_to_num(participant)))
    Clustering = np.array(Clustering)

    return Strength, Betweenness, Eigenvector, Clustering


temp_array = []
for i in range(8):
    temp = data[i, :, :]
    _, _, target_array = dyconnmap.analytic_signal(temp, fb=target_fb, fs=fs)
    temp_array.append(target_array)
input_data = np.array(temp_array)
del temp_array

adj_matrix = wpli_conn(data, target_fb, fs)

Strength, Betweenness, Eigenvector, Clustering = graph_metrics(adj_matrix)

np.save(output_dir + '_All_Strength_' + array_type, Strength)
np.save(output_dir + '_All_Betweenness_' + array_type, Betweenness)
np.save(output_dir + '_All_Eigenvector_' + array_type, Eigenvector)
np.save(output_dir + '_All_Clustering_' + array_type, Clustering)
np.save(output_dir + 'Master_adj_matrix_' + array_type, adj_matrix)

print('\n' + "EXECUTION TIME: " + str(time.time()-start))
