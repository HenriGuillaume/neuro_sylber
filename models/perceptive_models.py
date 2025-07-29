import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge
import os
import re
import pickle
from tqdm import tqdm
from utils import Subject, SplitDataset, split_data, open_pickle, CONFIG

# ECoG dataset
DS_DIR = CONFIG['data']['dataset_dir']
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

    def decode(self, dataset: SplitDataset):
        self.model.fit(dataset.train_X, dataset.train_y)
    
    def encode(self, dataset: SplitDataset):
        self.model.fit(dataset.train_y, dataset.train_X)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, dataset: SplitDataset):
        # only deconding score, will do decoding manually
        return self.model.score(dataset.test_X, dataset.test_y)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


def cross_validate_alpha(X, y, alphas, tmin, tmax, sfreq, n_splits=5):
    '''
    Important note: for encodeing task, just split X and y at input
    '''
    print(X.shape)
    print(y.shape)
    best_alpha = None
    best_score = -np.inf
    all_scores = {}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for alpha in alphas:
        fold_scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            trf_model = TRF(tmin, tmax, sfreq, alpha)

            trf_model.model.fit(X_train, y_train)
            score = trf_model.model.score(X_val, y_val)
            fold_scores.append(np.mean(score))  # average over output dims

        mean_score = np.mean(fold_scores)
        all_scores[alpha] = mean_score
        print(f"Alpha {alpha:.2e} | Mean CV Score: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha

    return best_alpha, all_scores


def save_outputs(preds, scores, filename, out_dir="TRF_outputs"):
    match = re.search(r'((sub\d{2}))_', filename)
    if not match:
        sub_name = 'unknown'
    else:
        sub_name = match.group(1)
    save_path = os.path.join(out_dir, sub_name)
    os.makedirs(save_path, exist_ok=True)

    # Save scores
    if scores is not None:
        print(f"Average score: {np.mean(scores):.4f}")
        scores_fname = filename + '_scores.pickle'
        np.save(os.path.join(save_path, scores_fname), scores)
        print(f"Saved scores to: {os.path.join(sub_folder, scores_fname)}")

    # Save predictions and targets
    preds_fname = filename + '_preds.pickle'
    np.save(os.path.join(save_path, preds_fname), preds)
    print(f"Saved predictions to: {os.path.join(save_path, preds_fname)}")


def run_TRF_pipeline(sub_num,
                hidden_states,
                train_ratio=0.5,
                tmin=-0.5,
                tmax=0.1,
                sfreq=50,
                alpha=1e6,
                test_ratio=None,
                pca=None,
                n_splits=5,
                weights=None,
                mode='clean',
                task='decode'): # or 'encode'
    '''
    test_ratio: float: ratio of the test set to compute the score on
    '''
    if pca is not None:
        pca = PCA(n_components=pca)
        HIDDEN_STATES = pca.fit_transform(HIDDEN_STATES)
    dataset = split_data(sub_num, hidden_states, train_ratio, val_ratio=0,
                         match_y=True, mode=mode)
    # sfreq is 50Hz if match_y, else 512

    # k-fold search for optimal alpha
    alpha_search_space = np.logspace(-1, 10, num=7)

    if task == 'decode':
        best_alpha, all_scores = cross_validate_alpha(dataset.train_X, dataset.train_y,
                                      alpha_search_space, tmin, tmax, sfreq, n_splits)
    elif task == 'encode':
        best_alpha, all_scores = cross_validate_alpha(dataset.train_y, dataset.train_X,
                                      alpha_search_space, tmin, tmax, sfreq, n_splits)
    
    trf_model = TRF(tmin, tmax, sfreq, best_alpha)
    if task == 'decode':
        trf_model.decode(dataset)
    elif task == 'encode':
        trf_model.encode(dataset)
    else:
        print('Task must be either encode or decode')
        return

    
    filename = f'sub{sub_num:02d}_task{task}_mode{mode}_r{train_ratio}_min{tmin}_max{tmax}_f{sfreq}_a{best_alpha:.2e}.pickle'
    #trf_model.save(filename)
    if task == 'decode':
        preds = trf_model.predict(dataset.test_X)
    else:
        preds = trf_model.predict(dataset.test_y)

    save_outputs(preds, scores=None, filename=filename)

    return trf_model, dataset

if __name__ == "__main__":
    import sys
    mode = 'clean'
    sub_num = int(sys.argv[1])
    task = sys.argv[2] # 'encode' or 'decode'
    mode = sys.argv[3]

    model, dataset = run_TRF_pipeline(
        sub_num=sub_num,
        hidden_states=HIDDEN_STATES,
        mode=mode,
        task=task  # Evaluate on full test set
    )
    print(f'optimal alpha: {scores['alpha']}')
