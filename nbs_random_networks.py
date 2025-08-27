import numpy as np
import bct

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
