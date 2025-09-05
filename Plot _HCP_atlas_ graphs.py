"""
A script for the plotting of HCP atlas graph into MNI space.
A number of changes are possible from here such as, communities and highlight_edges
See doc at: https://www.netplotbrain.org/
"""
import netplotbrain
import pandas as pd
import numpy as np

# # Get loc
# array = np.load(r'')

# # Plotting
# nodes = pd.DataFrame(data={'x': (array[:,0] - 128),
#                            'y': (array[:,1] - 150),
#                            'z': (array[:,2] - 100),
#                            })

# read the xyz file
file_path = "HCP_coordinates.csv"    # TODO: change to the right path
file_raw_data = pd.read_csv(file_path)

xyz_dataframe = pd.DataFrame({
    "x": list(file_raw_data["x-cog"]),
    "y": list(file_raw_data["y-cog"]),
    "z": list(file_raw_data["z-cog"]),
    "communites": [], # list of sig nodes, needs to be len = len(xyz)
})

nodes = xyz_dataframe

netplotbrain.plot(template='MNI152NLin2009cAsym',
                  nodes=nodes,
                  template_style='filled',
                  view=['LSR'],
                  arrowaxis=None,
                  node_scale=100)

