import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge
import os
import re
import pickle
from tqdm import tqdm
from utils import Subject, SplitDataset, split_data, open_pickle, CONFIG

# ECoG dataset
DS_DIR = CONFIG['dataset']['dataset_dir']
ECOGPREP_DIR = os.path.join(DS_DIR, 'derivatives/ecogprep/')
ECOGGQC_DIR = os.path.join(DS_DIR, 'derivatives/ecogqc/')
SUB_NAMES = [f'sub-0{i}' for i in range(10)]
WEIGHTS_DIR = './weights'

# sylber features of the podcast
SYLBER_FEAT_DIR = CONFIG['model']['sylber_outputs']
full_sylber_features = open_pickle(SYLBER_FEAT_DIR)

HIDDEN_STATES = full_sylber_features['hidden_states']


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


def run_pipeline(sub_num,
                hidden_states,
                train_ratio=0.5,
                tmin=-0.5,
                tmax=0.5,
                sfreq=256,
                alpha=1e6,
                test_ratio=None,
                pca=None,
                weights=None,
                mode='clean'):
    '''
    test_ratio: float: ratio of the test set to compute the score on
    weights: str: path to pickled weights
    '''
    if pca is not None:
        pca = PCA(n_components=pca)
        HIDDEN_STATES = pca.fit_transform(HIDDEN_STATES)
    dataset = split_data(sub_num, train_ratio, hidden_states, target_freq=sfreq, mode=mode)
    trf_model = TRF(tmin, tmax, sfreq, alpha)
    if weights is None:
        trf_model.fit(dataset)
    else:
        trf_model.load(weights)

    scores = None
    if test_ratio is not None:
        test_split_idx = int(test_ratio * dataset.test_X.shape[0])
        scores = trf_model.model.score(dataset.test_X[:test_split_idx], dataset.test_y[:test_split_idx])
        print(f"α: {alpha:.2e} Test scores (len={len(scores)}): mean={np.mean(scores):.4f}")

    filename = f'sub-{sub_num:02d}_r{train_ratio}_min{tmin}_max{tmax}_f{sfreq}_a{alpha:.2e}.pickle'
    trf_model.save(filename)
    print("Fitted model alpha:", trf_model.model.estimator.alpha)

    return trf_model, dataset, scores, {
        'sub_num': sub_num,
        'train_ratio': train_ratio,
        'tmin': tmin,
        'tmax': tmax,
        'sfreq': sfreq,
        'alpha': alpha,
        'mode': mode
    }


def save_outputs(preds, test_y, scores, params, out_dir="trf_outputs"):
    os.makedirs(out_dir, exist_ok=True)
    # predictions and targets
    preds_fname = f"sub-{params['sub_num']:02d}_mode{params['mode']}_preds_r{params['train_ratio']}_min{params['tmin']}_max{params['tmax']}_f{params['sfreq']}.npy"
    np.save(os.path.join(out_dir, preds_fname), (preds, test_y))
    print(f"Saved predictions to: {preds_fname}")


if __name__ == "__main__":
    sub_num = 1
    mode = 'hg'

    model, dataset, scores, params = run_pipeline(
        sub_num=sub_num,
        hidden_states=HIDDEN_STATES,
        mode=mode,
        test_ratio=1.0  # Evaluate on full test set
    )

    print("Predicting...")
    preds = model.predict(dataset.test_X)

    # Save all outputs
    save_outputs(preds, dataset.test_y, scores, params)
