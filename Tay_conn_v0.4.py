import numpy as np
import time
import mne
import os
import dyconnmap
from fontTools.ttx import process


# Parcellation of stc
def parcellation(array, labels):

    dec = {}
    for i in labels:
        index = labels.index(i)
        temp = str(labels[index])
        dec[temp] = array.magnitude().in_label(i).data 
    i = 0
    while True:
        temp = dec[str(labels[i])]
        dec[str(labels[i])] = np.mean(temp, axis=0)
        i += 1
        if i >= 360:
            break
    list_1 = [i for i in dec.values()]
    list_2 = [np.expand_dims(i, axis=0) for i in list_1]
    temp = np.concatenate(list_2)
    return temp


def prep_data(input_dir, target_fb, fs):
    data = []
    base_file_names = []
    
    labels = mne.read_labels_from_annot("fsaverage", "HCPMMP1", "both")
    del labels[0]
    del labels[0]
    
    for file in os.listdir(input_dir):
        stc_path = os.path.join(input_dir, file)
        base_file_names.append(os.path.basename(stc_path))
        target_stc = mne.read_source_estimate(stc_path)
        target_array = parcellation(target_stc, labels)
        target_array = target_array[:, 10:14490]    # crop the first and last 10 time-points, likely unneeded
        _, _, target_array = dyconnmap.analytic_signal(target_array.T, fb=target_fb, fs=fs)
        data.append(target_array)
    return np.stack(data, axis=0), base_file_names


# wPLI
def wpli_conn(array, base_file_names, target_fb, fs, save_base):   # (participants, roi's, time_points)
    Master_adj_matrix = []
    for i in range(len(base_file_names)):
        input_data = array[i, :, :]
        adj_matrix = dyconnmap.fc.wpli(input_data, fs=fs, fb=target_fb)
        Master_adj_matrix.append(adj_matrix)
        name = base_file_names[i]
        np.save(save_base + name + '_conn_matrix', adj_matrix)
    return np.array(Master_adj_matrix)


# Main
start = time.time()
saving_path = r'G:\Vixen\adj_arrays\_'
file_path = r'G:\Vixen\raw_stc'


Master_array, file_names = prep_data(file_path, [8, 13], 250)
Master_adj_matrix = wpli_conn(Master_array, file_names, [8, 13], 250, saving_path)
print('\n' + "EXECUTION TIME: " + str(time.time() - start) + " sec")










labels = mne.read_labels_from_annot("fsaverage", "HCPMMP1", "both")
del labels[0]
del labels[0]


import multiprocessing

def fun():
    target_fb = [8, 13]
    fs = 250

    data = mne.read_source_estimate(r's13_stc_h2-stc.h5')

    start = time.time()
    target_array = parcellation(data, labels)
    print('\n' + "EXECUTION TIME: " + str(time.time() - start) + " sec")

    target_array = target_array[:, 10:14490]  # crop the first and last 10 time-points, likely unneeded
    _, _, target_array = dyconnmap.analytic_signal(target_array.T, fb=target_fb, fs=fs)

    start = time.time()
    adj_matrix = dyconnmap.fc.wpli(target_array.T, fs=fs, fb=target_fb)
    print('\n' + "EXECUTION TIME: " + str(time.time() - start) + " sec")

    np.save(saving_path + 's13_stc_h2' + '_adj_matrix', adj_matrix)


start = time.time()
process = multiprocessing.Process(target=fun,args=('process1'))
process.start()
process.join()
print('\n' + "EXECUTION TIME: " + str(time.time() - start) + " sec")

