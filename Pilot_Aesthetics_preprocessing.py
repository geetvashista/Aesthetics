import mne
import numpy as np
import matplotlib

# Do we want to epoch? If so, set to True.
epoch = True
tmin = 1    # Set as wanted if epoch is True
tmax = 60    # Set as wanted if epoch is True

# Set back ends of viz
matplotlib.use('Qt5Agg')

# Set file paths
data_file = r'C:\Users\em17531\Desktop\Tamar_data\p01\eeg\neuroaesthetics_p01.vhdr'
captrak_file = r'C:\Users\em17531\Desktop\Tamar_data\p01\captrak\neuroaesthetics_p01.bvct'

def preprocessing(data_file, captrak_file):
    # load data
    raw = mne.io.read_raw_brainvision(data_file)

    # set background info
    raw.info['line_freq'] = 50

    # Set events
    events, event_id = mne.events_from_annotations(raw)

    # Mark bad channels and check with plot
    raw.load_data()
    raw.plot(block=True)

    # notch filter
    raw_notch = raw.copy().notch_filter(np.arange(50, 500, 50))

    # high pass filter set to 1 Hz (to 100 Hz)
    raw_filtered = raw_notch.copy().filter(l_freq=1, h_freq=100)

    # Virtual reference (average referencing here)
    raw_avg_ref = raw_filtered.copy().set_eeg_reference(ref_channels='average')

    # Interpolation of bad channels
    raw_interp = raw_avg_ref.interpolate_bads(reset_bads=False)

    # Setting up a Montage
    montage = mne.channels.read_dig_captrak(captrak_file)
    raw_with_montage = raw_interp.set_montage(montage)

    # Independent Components Analysis (ICA) for artifact removal
    ica = mne.preprocessing.ICA(n_components=32, random_state=0)
    ica.fit(raw_with_montage)
    ica.plot_components()
    ica.plot_sources(raw_with_montage, title='EEG sources estimated by ICA', block=True)
    clean_data = ica.apply(raw_with_montage.copy())
    clean_data.plot()

    return clean_data, events, event_id


def epoch_data(clean_data, events, event_ids, tmin, tmax):
    # epoch data
    picks = mne.pick_types(clean_data.info,
                           meg=False,
                           eeg=True,
                           stim=False,
                           eog=False,
                           )

    epochs = mne.Epochs(clean_data,
                        events=events,
                        event_id=event_ids,
                        tmin=tmin,
                        tmax=tmax,
                        baseline=None,
                        picks=picks)

    return epochs

# "__main__"
# Preprocessing
clean_data, events, event_ids = preprocessing(data_file=data_file, captrak_file=captrak_file)

# epoch
if epoch:
    epochs = epoch_data(clean_data, events, event_ids, tmin, tmax)
    # downsample data from to 250 Hz
    epochs.load_data()
    epochs = epochs.resample(250, npad="auto")

