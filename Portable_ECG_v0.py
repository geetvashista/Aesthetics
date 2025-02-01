import pandas as pd
import matplotlib
import neurokit2 as nk
from matplotlib import pyplot as plt

# set back end
matplotlib.use('TkAgg')

# file name and directory
directory_of_file = '/home/students/Ddrive_2TB/Geet/Tamar_data/'
file_name = 'tay_test_1.csv'

# load data
raw_ecg = pd.read_csv(directory_of_file + file_name)

# construct signale
ecg_signal = list(raw_ecg['T3 (uV)'] - raw_ecg['C3 (uV)'])

# Viz data
# TODO: we need to find out what this actually is. Using 250 as a proxy.
def plot_signal(data):
        plt.plot(data)
        plt.show()

plot_signal(ecg_signal[12*250:15*250])
# nk.signal_plot(ecg_signal)      # viz with neurokit

# filtering
temp, _ = nk.ecg_process(ecg_signal, sampling_rate=250)
data = temp['ECG_Clean']
plot_signal(data[12*250:15*250])

# Idea: the signal is still a little noisy, let just nuke it to hell with a 
# low-pass filter at like 20Hz?
