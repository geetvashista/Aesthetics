import numpy as np
import networkx as nx
import bct
import seaborn as sns
import scipy.stats as stats
import matplotlib

# Setting some backend perameters
matplotlib.use('TkAgg')

# Import data
raw_array_file_path = r'C:\Users\em17531\Desktop\adj_matrices\_s10_stc_h1_adj_matrix.npy'

raw_array = np.load(raw_array_file_path)
del raw_array_file_path

# Cleaning self edges
g = nx.from_numpy_array(raw_array)
g.remove_edges_from(list(nx.selfloop_edges(g)))
adj_array = nx.to_numpy_array(g)

