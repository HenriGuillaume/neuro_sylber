import os
import re
import pickle
import numpy as np
import mne
import yaml
import pandas as pd
import h5py

with open("../config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)


DS_DIR = '../' + CONFIG['data']['dataset_dir']
ECOGPREP_DIR = os.path.join(DS_DIR, 'derivatives/ecogprep/')
ECOGGQC_DIR = os.path.join(DS_DIR, 'derivatives/ecogqc/')
SYLBER_OUTPUTS = CONFIG['model']['sylber']['outputs']

# reding h5 files
def get_h5_layer(outputs_pth=SYLBER_OUTPUTS, layer_num=9):
    with h5py.File(outputs_pth, "r") as h5f:
        hidden_states = h5f["all_hidden_states"]
        return hidden_states[:, layer_num, :][:]

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
        self.paths = {
            mode:os.path.join(ecogprep_dir, self.sub_name, 'ieeg',
                    self.sub_name + '_task-podcast_desc-' + mode + '_ieeg.fif')
            for mode in ('alpha', 'gamma', 'theta', 'beta', 'highgamma')
        }
        self.paths['clean'] = os.path.join(ecogprep_dir ,self.sub_name, 'ieeg', 
                                      self.sub_name + '_task-podcast_ieeg.fif')
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

    def __add__(self, other):
        if not isinstance(other, SplitDataset):
            raise TypeError("Can only add SplitDataset to SplitDataset")
        
        return SplitDataset(
            train_X=np.concatenate([self.train_X, other.train_X], axis=0),
            val_X=np.concatenate([self.val_X, other.val_X], axis=0),
            test_X=np.concatenate([self.test_X, other.test_X], axis=0),
            train_y=np.concatenate([self.train_y, other.train_y], axis=0),
            val_y=np.concatenate([self.val_y, other.val_y], axis=0),
            test_y=np.concatenate([self.test_y, other.test_y], axis=0),
        )


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
               match_X=False,
               match_y=False,
               match_x_func=stepwise_resample,
               match_y_func=average_pool,
               mode='hg'):
    '''
    Match input features (ECoG) sr to output features (hidden_states) sr
    or the inverse, then split according to ratio
    '''
    sub = Subject(sub_num) # maybe put this as a getter of Subject class
    raw = mne.io.read_raw_fif(sub.paths[mode])

    X = raw.get_data().T
    print(f'Reading raw file of size {X.shape}')

    num_samples_X = X.shape[0]
    num_samples_y = y.shape[0]
    min_num_samples = min(num_samples_X, num_samples_y)
    if match_X:
        y = match_x_func(y, num_samples_X)
        y = y[:min_num_samples]
        X = X[:min_num_samples]
    elif match_y:
        X = match_y_func(X, num_samples_y)
        y = y[:min_num_samples]
        X = X[:min_num_samples]

    # Compute split indices
    num_samples_X = X.shape[0] # they may have changed
    num_samples_y = y.shape[0]
    train_end_X = int(train_ratio * num_samples_X)
    train_end_y = int(train_ratio * num_samples_y)
    val_end_X = train_end_X + int(val_ratio * num_samples_X)
    val_end_y = train_end_y + int(val_ratio * num_samples_y)

    return SplitDataset(
        train_X=X[:train_end_X],
        val_X=X[train_end_X:val_end_X],
        test_X=X[val_end_X:],
        train_y=y[:train_end_y],
        val_y=y[train_end_y:val_end_y],
        test_y=y[val_end_y:]
    )


def open_pickle(pth):
    with open(pth, 'rb') as handle:
        b = pickle.load(handle)
    return b


def tsv_to_VAD(tsv_path, sr=50, total_duration=1800.0):
    '''
    Outputs a binary array of the total duration (30mins in 
    our case dor the podcast dataset), at the given sampling
    rate.
    '''
    samples_len = int(total_duration * sr)
    df = pd.read_csv(tsv_path, sep="\t")
    sample_rate = samples_len / total_duration
    vad = np.zeros(samples_len, dtype=np.uint8)
    for _, row in df.iterrows():
        start_sample = int(row['start'] * sample_rate)
        end_sample = int(row['end'] * sample_rate)
        start_sample = max(0, start_sample)
        end_sample = min(samples_len, end_sample)
        vad[start_sample:end_sample] = 1

    return vad


def save_predictions(predictions,
                     targets, # set to None to avoid saving same gt everytime
                     model_id,
                     out_dir=CONFIG['outputs']):
    sub_match = re.search(r'(sub\d{2})', model_id)
    if not sub_match:
        sub_name = 'unknown'
    else:
        sub_name = sub_match.group(1)
    model_match = re.search(r'TRF', model_id)
    if not model_match:
        model_name = 'transformer'
    else:
        model_name= 'TRF'
    save_path = os.path.join(out_dir, model_name, sub_name)
    os.makedirs(save_path, exist_ok=True)

    # Save files
    preds_fname = f"predictions_{model_id}.npy"
    gt_fname = f"ground_truth_{model_id}.npy"

    np.save(os.path.join(save_path, preds_fname), predictions)
    print(f"Saved predictions to: {os.path.join(out_dir, preds_fname)}")
    if targets is not None:
        np.save(os.path.join(save_path, gt_fname), targets)
        print(f"Saved ground truth to: {os.path.join(out_dir, gt_fname)}")

