import numpy as np
import bct
import netplotbrain
import matplotlib

matplotlib.use("TkAgg")

array = []
for _ in range(10):
    temp = np.random.rand(360, 360)
    array.append(temp)
del temp
array = np.array(array)


y_array = []
for _ in range(10):
    temp = np.random.rand(360, 360)
    y_array.append(temp)
del temp
y_array = np.array(y_array)

pvals, adj, _ = bct.nbs_bct(array.T, y_array.T, thresh = 3)



# Get loc
array = adj

# Plotting
nodes = pd.DataFrame(data={'x': (array[:,0] - 128),
                           'y': (array[:,1] - 150),
                           'z': (array[:,2] - 100),
                           })

netplotbrain.plot(template='MNI152NLin2009cAsym',
                  nodes=nodes,
                  template_style='filled',
                  view=['LSR'],
                  arrowaxis=None,
                  node_scale=100)
