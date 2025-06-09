# This is script to calculate connectivity using raw np arrays,

# each assumed to be in the shape (channels/ROI's, time points)



import numpy as np

import time

import os

import dyconnmap

import bct

start = time.time()



def operations_to_perform():    # Change as desired

    Cal_wpli = True

    Cal_graph_metrics = True

    return Cal_wpli, Cal_graph_metrics





def setup_dependencies():    # Change as desired

    input_dir = r'C:\Users\em17531\Desktop\OPM_MEG\data'     # The directory containing files of interest

    output_dir = r'C:\Users\em17531\Desktop\OPM_MEG\Graph_output\Beta'    # The output directory

    array_type = 'run-002'     # This is the type of array that will be loaded for further calculations, eg. "participant_conversation_alpha"

    target_fb = [13, 30]  # The target frequency band     TODO: set this up so variable can be a this can be list of bands and each generates it's own adjacency matrix

    band = '_beta_'

    fs = 1200    # The sampling frequency

    saving = True

    os.makedirs(output_dir, exist_ok=True)

    return input_dir, output_dir, array_type, fs, target_fb, saving, band





def prep_data(input_dir, array_type, target_fb, fs):

    data = []

    for folder in os.listdir(input_dir):

        for file in os.listdir(os.path.join(input_dir, folder)):

            if array_type in os.path.basename(file):

                target_array = np.load(os.path.join(input_dir, folder, file))

                target_array = target_array[0:606000, :]

                _, _, target_array = dyconnmap.analytic_signal(target_array.T, fb=target_fb, fs=fs)

                data.append(target_array)

    return np.stack(data, axis=0)





def wpli_conn(array, target_fb, fs):   # (participants, roi's, time_points)

    adj_matrix = []

    adj_matrix.append(dyconnmap.fc.wpli(array, fs=fs, fb=target_fb))

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





def main():     # TODO: put in the elif statements

    # Prep

    Cal_wpli, Cal_graph_metrics = operations_to_perform()

    input_dir, output_dir, array_type, fs, target_fb, saving, band = setup_dependencies()

    data = prep_data(input_dir, array_type, target_fb, fs)



    # Core functions

    if Cal_wpli:

        adj_matrix = wpli_conn(data, target_fb, fs)

    if Cal_graph_metrics:

        Strength, Betweenness, Eigenvector, Clustering = graph_metrics(adj_matrix)



    # Saving

    if saving:

        np.save(output_dir + '_All_Strength_' + band + array_type, Strength)

        np.save(output_dir + '_All_Betweenness_' + band + array_type, Betweenness)

        np.save(output_dir + '_All_Eigenvector_' + band + array_type, Eigenvector)

        np.save(output_dir + '_All_Clustering_' + band + array_type, Clustering)

        np.save(output_dir + '_Master_adj_matrix_' + band + array_type, adj_matrix)





if __name__ == "__main__":

    main()



print('\n' + "EXECUTION TIME: " + str(time.time()-start))
