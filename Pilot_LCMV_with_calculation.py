import mne
import pathlib
from mne.beamformer import apply_lcmv_epochs, make_lcmv
import time
from mne.datasets import fetch_fsaverage

# Path to epoch file
epoch_path = r''

# Loading data
epochs = mne.read_epochs(epoch_path)

def LCMV(epochs):
    # Start a timer. This isn't really necessary
    start = time.time()

    # Download needed fsaverage files for a forward model
    fs_dir = pathlib.Path(fetch_fsaverage())
    subjects_dir = fs_dir.parent
    subject = "fsaverage"
    trans = "fsaverage"  # MNE has a built-in fsaverage transformation
    src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
    bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"

    # Calculate covariance matrice NOTE: noise matrix not needed for this study
    data_cov = mne.compute_covariance(epochs, tmin=0, tmax=1, method="empirical")

    # Set up forward model using a free surfer average
    fwd = mne.make_forward_solution(
        epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None
    )

    # Estimate filter weights
    filters = make_lcmv(
        epochs.info,
        fwd,
        data_cov,
        reg=0.05,
        pick_ori="max-power",
        weight_norm="unit-noise-gain",
        rank=None,
    )

    # Apply LCMV
    epochs.set_eeg_reference(projection=True)
    stc = apply_lcmv_epochs(epochs, filters)

    end = time.time()
    print('\n')
    print("The time of execution of above program is :",
          (end - start), "_s")

    return stc

# "__main__"

stc = LCMV(epochs)
