import os
import re
import pickle
import numpy as np
import mne
import yaml

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)


DS_DIR = CONFIG['data']['dataset_dir']
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
    def __init__(self, train_X, val_X, test_X, train_y, val_y, test_y):
        self.train_X = train_X
        self.val_X = val_X
        self.test_X = test_X
        self.train_y = train_y
        self.val_y = val_y
        self.test_y = test_y

    def summary(self):
        return {
            'train_X': self.train_X.shape,
            'val_X': self.val_X.shape,
            'test_X': self.test_X.shape,
            'train_y': self.train_y.shape,
            'val_y': self.val_y.shape,
            'test_y': self.test_y.shape
        }


def stepwise_resample(data, target_len):
    idx = np.floor(np.linspace(0, data.shape[0], target_len, endpoint=False)).astype(int)
    idx = np.clip(idx, 0, data.shape[0] - 1)
    return data[idx]


def average_pool(high_rate_signal, target_len):
    high_T, channels = high_rate_signal.shape
    stride = high_T // target_len

    # Truncate the signal to match an exact multiple
    truncated_len = target_len * stride
    x = high_rate_signal[:truncated_len, :]

    # Reshape
    x = x.reshape(target_len, stride, channels)

    # average over stride dimension
    x = x.mean(axis=1)

    return x


def split_data(sub_num,
               y,
               train_ratio=0.8,
               val_ratio=0.1,
               match_y=True,
               match_x_func=stepwise_resample,
               match_y_func=average_pool,
               mode='hg'):
    '''
    Match input features (ECoG) sr to output features (hidden_states) sr
    or the inverse, then split according to ratio
    '''
    sub = Subject(sub_num)
    if mode == 'hg':
        raw = mne.io.read_raw_fif(sub.hg_path)
    else:
        raw = mne.io.read_raw_fif(sub.clean_path)

    X = raw.get_data().T

    if not match_y:
        num_samples = X.shape[0]
        y = match_x_func(y, num_samples)
    else:
        num_samples = y.shape[0]
        X = match_y_func(X, num_samples)

    assert num_samples == y.shape[0], 'Shape mismatch'

    # Compute split indices
    train_end = int(train_ratio * num_samples)
    val_end = train_end + int(val_ratio * num_samples)

    return SplitDataset(
        train_X=X[:train_end],
        val_X=X[train_end:val_end],
        test_X=X[val_end:],
        train_y=y[:train_end],
        val_y=y[train_end:val_end],
        test_y=y[val_end:]
    )

def open_pickle(pth):
    with open(pth, 'rb') as handle:
        b = pickle.load(handle)
    return b
