import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge
import os
import re
import pickle
from utils import Subject, SplitDataset, TRF, split_data, open_pickle
from tqdm import tqdm

# ECoG dataset
DS_DIR = './ds005574-1.0.2/'
ECOGPREP_DIR = os.path.join(DS_DIR, 'derivatives/ecogprep/')
ECOGGQC_DIR = os.path.join(DS_DIR, 'derivatives/ecogqc/')
SUB_NAMES = [f'sub-0{i}' for i in range(10)]
WEIGHTS_DIR = './weights'

# sylber features of the podcast
SYLBER_FEAT_DIR = './pickled_podcast/'
full_sylber_features = open_pickle(os.path.join(SYLBER_FEAT_DIR,
'outputs.pkl'))

HIDDEN_STATES = full_sylber_features['hidden_states']
SYLBER_PCA = 0 # set to 0 for no PCA
if SYLBER_PCA:
    pca = PCA(n_components=SYLBER_PCA)
    HIDDEN_STATES = pca.fit_transform(HIDDEN_STATES)


def run_pipeline(sub_num,
                hidden_states,
                train_ratio=0.8,
                tmin=-0.5,
                tmax=0.5,
                sfreq=256,
                alpha=1.0,
                test_ratio=None,
                weights=None,
                mode='clean'):
    '''
    test_ratio: float: ratio of the test set to compute the score on
    weights: str: path to pickled weights
    '''
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

alphas = np.logspace(np.log10(0.1), np.log10(1e8), num=8)
opt_alpha = alphas[6]
sub_num = 1

model, dataset, scores, params = run_pipeline(sub_num, HIDDEN_STATES, alpha=opt_alpha, mode='clean', weights='weights/')

# Save scores
scores_fname = f"sub-{params['sub_num']:02d}_mode{params['mode']}_scores_r{params['train_ratio']}_min{params['tmin']}_max{params['tmax']}_f{params['sfreq']}.npy"
np.save(scores_fname, scores)

# Run inference and save predictions
print('Predicting...')
preds = model.predict(dataset.test_X)
preds_fname = f"sub-{params['sub_num']:02d}_mode{params['mode']}_scores_r{params['train_ratio']}_min{params['tmin']}_max{params['tmax']}_f{params['sfreq']}_preds.npy"
np.save(preds_fname, (preds, dataset.test_y))
