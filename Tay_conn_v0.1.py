import numpy as np
import time
import mne
import os
import dyconnmap
# start = time.time()


def prep_data(input_dir, target_fb, fs):
    data = []
    base_file_names = []
    for file in os.listdir(input_dir):
        stc_path = os.path.join(input_dir, file)
        base_file_names.append(os.path.basename(stc_path))
        target_array = mne.read_source_estimate(stc_path)
        target_array = target_array.magnitude()
        target_array = target_array.data
        target_array = target_array[:, 10:14490]
        _, _, target_array = dyconnmap.analytic_signal(target_array.T, fb=target_fb, fs=fs)
        data.append(target_array)
    return np.stack(data, axis=0), base_file_names


# wPLI
def wpli_conn(array, base_file_names, target_fb, fs, save_base):   # (participants, roi's, time_points)
    Master_adj_matrix = []
    for i in range(len(base_file_names)):
        input_data = array[i, :, :]
        adj_matrix = dyconnmap.fc.wpli(input_data.T, fs=fs, fb=target_fb)
        Master_adj_matrix.append(adj_matrix)
        name = base_file_names[i]
        np.save(save_base + name + '_conn_matrix', adj_matrix)
    return np.array(Master_adj_matrix)


# Main
start = time.time()
saving_path = r'G:\Vixen\adj_arrays\_'
file_path = r'G:\Vixen\raw_stc'

# Master_array, file_names = prep_data(file_path, [8, 13], 250)
file_names = ['s10_stc_h1-stc.h5',
              's10_stc_h2-stc.h5',
              's10_stc_l1-stc.h5',
              's10_stc_l2-stc.h5',
              's13_stc_h1-stc.h5',
              's13_stc_h2-stc.h5',
              's13_stc_l1-stc.h5',
              's13_stc_l2-stc.h5']

Master_array = np.load('Tay_master_array.npy')

Master_adj_matrix = wpli_conn(Master_array, file_names, [8, 13], 250, saving_path)
print('\n' + "EXECUTION TIME: " + str(time.time() - start) + " sec")


