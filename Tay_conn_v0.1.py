import numpy as np
import time
import mne
import os
import dyconnmap
import bct
start = time.time()

# Acquire and prep data TO DO: adjust for Tay's needs
def prep_data(input_dir, target_fb, fs):
    data = []
    base_file_names = []
    for file in os.listdir(input_dir):
        stc_path = os.path.join(input_dir, folder, file)
        base_file_names.append(os.path.basename(stc_path))
        target_array = np.load(stc_path)   # <-- Let's adjust this to take an actual stc
        # rather than an array

        target_array = target_array.magnitude()
        target_array = target_array.get_data()

        target_array = target_array[:, :]
        _, _, target_array = dyconnmap.analytic_signal(target_array.T, fb=target_fb, fs=fs)
        data.append(target_array)
    return np.stack(data, axis=0), base_file_names


# wPLI  <-- Not sure if we need to adjust this as well to work for a single participant
def wpli_conn(array, target_fb, fs):   # (participants, roi's, time_points)
    adj_matrix = []
    for participant in array:
        adj_matrix.append(dyconnmap.fc.wpli(participant, fs=fs, fb=target_fb))
    return np.array(adj_matrix)

# Main
file_path = ''

temp = mne.read_source_estimate(file_path)
temp = temp.magnitude()
