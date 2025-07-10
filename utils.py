import os
import re
import pickle
import numpy as np
import mne
from mne.decoding import ReceptiveField
from mne.decoding import TimeDelayingRidge


DS_DIR = './ds005574-1.0.2/'
ECOGPREP_DIR = os.path.join(DS_DIR, 'derivatives/ecogprep/')
ECOGGQC_DIR = os.path.join(DS_DIR, 'derivatives/ecogqc/')

class Subject:
    def __init__(self, 
        sub_num,
        ecogprep_dir=ECOGPREP_DIR,
        ecogqc_dir=ECOGGQC_DIR):
        '''
        Store subject data
        '''
        self.sub_name = f'sub-0{sub_num}'
        # BRAIN DATA
        # high gamma features
        hg_suffix = '_task-podcast_desc-highgamma_ieeg.fif'
        self.hg_path = os.path.join(ecogprep_dir, self.sub_name, 
        'ieeg', self.sub_name + hg_suffix)
        # cleaned data, not HG extraction
        clean_suffix = '_task-podcast_ieeg.fif'
        self.clean_path = os.path.join(ecogprep_dir, self.sub_name, 
        'ieeg', self.sub_name + clean_suffix)
        # info about electrodes
        self.channels_log = os.path.join(ecogqc_dir, self.sub_name, 
        'mark_channels.log')


class SplitDataset:
    def __init__(self, train_X, test_X, train_y, test_y):
        self.train_X = train_X
        self.test_X = test_X
        self.train_y = train_y
        self.test_y = test_y

    def summary(self):
        return {
            'train_X': self.train_X.shape,
            'test_X': self.test_X.shape,
            'train_y': self.train_y.shape,
            'test_y': self.test_y.shape
        }


class TRF:
    def __init__(self, tmin=-0.2, tmax=1.0, sfreq=256, alpha=1.0, weights=None):
        self.tmin = tmin
        self.tmax = tmax
        self.sfreq = sfreq
        self.alpha = alpha
        self.model = ReceptiveField(
            tmin=tmin,
            tmax=tmax,
            sfreq=sfreq,
            scoring='corrcoef',
            estimator=TimeDelayingRidge(tmin, tmax, sfreq, reg_type="ridge", alpha=alpha),
            n_jobs='cuda'
        )
        if weights is not None:
            self.load(weights)

    def fit(self, dataset: SplitDataset):
        self.model.fit(dataset.train_X, dataset.train_y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, dataset: SplitDataset):
        return self.model.score(dataset.test_X, dataset.test_y)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


def stepwise_resample(data, target_len):
    idx = np.floor(np.linspace(0, data.shape[0], target_len, endpoint=False)).astype(int)
    idx = np.clip(idx, 0, data.shape[0] - 1)
    return data[idx]


def split_data(sub_num, train_ratio=0.5, hidden_states=None, target_freq=256):
    sub = Subject(sub_num)
    raw = mne.io.read_raw_fif(sub.hg_path)
    raw.resample(target_freq)

    hg_data = raw.get_data().T.astype(np.float32)
    num_samples = hg_data.shape[0]
    hidden_states = stepwise_resample(hidden_states, num_samples).astype(np.float32)

    assert num_samples == hidden_states.shape[0], 'Shape mismatch'
    split_idx = int(train_ratio * num_samples)

    return SplitDataset(
        train_X=hg_data[:split_idx],
        test_X=hg_data[split_idx:],
        train_y=hidden_states[:split_idx],
        test_y=hidden_states[split_idx:]
    )

def open_pickle(pth):
    with open(pth, 'rb') as handle:
        b = pickle.load(handle)
    return b    