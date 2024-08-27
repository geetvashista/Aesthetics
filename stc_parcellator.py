import mne
import numpy as np

# Parcellation for sham
labels = mne.read_labels_from_annot("fsaverage", "HCPMMP1", "both")
del labels[0]
del labels[0]

total = []
for i in range(8):
    dec = {}
    for k in labels:
        index = labels.index(k)
        temp = str(labels[index])
        dec[temp] = stc[i].in_label(k).data  # change here from buds to sham as needed

    j = 0
    while True:
        temp = dec[str(labels[j])]
        dec[str(labels[j])] = np.mean(temp, axis=0)
        j += 1
        if j >= 360:
            break

    list = [i for i in dec.values()]
    list_2 = [np.expand_dims(i, axis=0) for i in list]
    temp_sham = np.concatenate(list_2)
    total.append(temp_sham)

stc_par = np.array(total)
np.save('p01_stc_parcellated_array', stc_par)