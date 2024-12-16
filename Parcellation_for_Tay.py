import mne
import numpy as np
import os
import xml.etree.ElementTree as ET
import netplotbrain
import pandas as pd
import time


data_path = r'G:\Vixen\averages\s13_h1_av-stc.h5'

data = mne.read_source_estimate(data_path)

labels = mne.read_labels_from_annot("fsaverage", "HCPMMP1", "both")
del labels[0]
del labels[0]

dec = {}
for i in labels:
    index = labels.index(i)
    temp = str(labels[index])
    dec[temp] = data.magnitude().in_label(i).data  # change here from buds to sham as needed
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

sort = np.argsort(temp, axis = 0)
sort_list = list(sort)

ordered_areas = []
[ordered_areas.append(labels[int(i)]) for i in sort_list]
print(ordered_areas)

#---------------------------------------------------------------------------------------------------------#
# Plotting
# TO DO: get this working

start = time.time()
def parse_xml(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    coordinates = []

    # Iterate over 'person' elements
    data_element = root.find('data')
    if data_element is not None:
        for label in data_element.findall('label'):
            # Extract attributes
            x = label.get('x')
            y = label.get('y')
            z = label.get('z')
            coordinates.append((float(x), float(y), float(z)))
    return coordinates


file_path = r'C:\Users\em17531\Desktop\Atlas\HCP-Multi-Modal-Parcellation-1.0.xml'  # Change this as needed
coordinates = parse_xml(file_path)
del coordinates[0]
xyz_loc_array = np.stack(coordinates)

array = xyz_loc_array
# del xyz_loc_array


# Invert function
def inverter(coor):
    inverted_coordinates = [(-x, -y, z) for (x, y, z) in coor]
    return inverted_coordinates
array = np.array(inverter(array))

# Make a list that marks only the first 10 positions as a node of interest
zero_list = [0] * 360

target = sort_list[:10]

for i in target:
    zero_list[int(i)] = 1

nodes = pd.DataFrame(data={'x': (array[:,0] + 100),
                           'y': (array[:,1]),
                           'z': (array[:,2] + 120),
                           'communities': zero_list
                           })

import matplotlib
# Set back ends of viz
matplotlib.use('Qt5Agg')

netplotbrain.plot(template='MNI152NLin2009cAsym',
                  nodes=nodes,
                  node_color='communities',
                  node_size=20,
                  node_type='circles',
                  view=['LSR'],
                  template_style='glass',
                  title='s10 top 10 highest power areas')

print('\n' + "EXECUTION TIME: " + str(time.time() - start) + " sec")




# nodes = pd.DataFrame(data={'x': (array[:,0] - 128),
#                            'y': (array[:,1] - 150),
#                            'z': (array[:,2] - 100),
#                            'communities': zero_list
#                            })
#
# netplotbrain.plot(template='MNI152NLin2009cAsym',
#                   nodes=nodes,
#                   node_color='communities',
#                   node_size=20,
#                   node_type='circles',
#                   view=['I'],
#                   template_style='glass',
#                   title='s13 top 10 highest power areas')
