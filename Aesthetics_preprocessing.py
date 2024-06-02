# Author: Monkey (gv.3721@hotmail.co.nz)
# Preprocessing for Vixen

import mne
import numpy as np
import scipy
import pandas as pd
import matplotlib

# Set back ends of viz

matplotlib.use('Qt5Agg')


# Set participant parameters

sfreq = 1000
tmin = 0
tmax = 1
freqs = [10]


# Set file paths

data_file = r'C:\Users\em17531\Desktop\Tamar_data\Jas_matlab_arrays\Raw_array.mat'
ch_file = r'C:\Users\em17531\Desktop\Tamar_data\Jas_matlab_arrays\ch_names.csv'
events_file = r'C:\Users\em17531\Desktop\Tamar_data\Jas_matlab_arrays\events_file.csv'


# Importing data

temp = scipy.io.loadmat(data_file)
Data_array = temp['Raw_array']
del temp
del data_file

ch_names_df = pd.read_csv(ch_file)
ch_names = ch_names_df['label'].tolist()
del ch_names_df
del ch_file

temp = pd.read_csv(events_file)
events_array = np.array(temp['latency'])
del temp
del events_file


# Creating raw mne object

info = mne.create_info(ch_names, sfreq, 'eeg')
raw = mne.io.RawArray(Data_array, info)

del info
del sfreq
del ch_names
del Data_array


# set background info

raw.info['line_freq'] = 50

# Set Annotations and events
an = mne.Annotations((events_array/1000), 1, 'good')
raw_an = raw.set_annotations(an)
del raw


# Mark bad chs

raw_an.plot(block=True)


   ### Cleaning data ###

# notch filter
raw_notch = raw_an.copy().notch_filter(np.arange(50, 500, 50))


# high pass filter set to 1 Hz (to 100 Hz)
raw_filtered = raw_notch.copy().filter(l_freq=1, h_freq=100)


# Virtual reference (average referencing here)
raw_avg_ref = raw_filtered.copy().set_eeg_reference(ref_channels='average')


# Interpolation of bad channels
raw_interp = raw_avg_ref.interpolate_bads(reset_bads=False)


# downsample data from to 250 Hz
raw_downsampled = raw_interp.resample(250, npad="auto")


# cleaning up for memory
del raw_notch
del raw_filtered
del raw_interp
del raw_an


# Setting up a Montage

montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
new_ch_names = montage.ch_names
old_names = raw_downsampled.ch_names
dic = dict(zip(old_names,new_ch_names))
raw_renamed = raw_downsampled.rename_channels(dic)
raw_renamed = raw_renamed.set_montage(montage)


# Independent Components Analysis (ICA) for artifact removal

ica = mne.preprocessing.ICA(n_components=32, random_state=0)
ica.fit(raw_renamed)
ica.plot_sources(raw_renamed, title='EEG sources estimated by ICA')
ica.plot_components()
raw_ica = ica.apply(raw_renamed.copy())
raw_ica.plot(block=True)


# extract events
events, event_id = mne.events_from_annotations(raw_ica)


# epoch data
picks = mne.pick_types(raw_ica.info,
                       meg=False,
                       eeg=True,
                       stim=False,
                       eog=False,
                       exclude='bads')

epochs = mne.Epochs(raw_ica,
                    events=events,
                    event_id=event_id,
                    tmin=tmin,
                    tmax=tmax,
                    proj=True,
                    picks=picks,
                    baseline=None,
                    preload=True)
