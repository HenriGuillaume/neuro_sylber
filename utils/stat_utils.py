import numpy as np
import scipy as sp
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine

def cutoff_or_pad(X, target_len):
    if len(X) == target_len:
        return X
    if len(X) > target_len:
        return X[:target_len]
    else:
        return np.pad(X, target_len - len(X))


def componentwise_spearman(X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    assert X.shape == Y.shape, "Signals must have the same shape"
    
    T, D = X.shape
    coeffs = np.zeros(T)
    
    for t in range(T):
        coeff, _ = spearmanr(X[t, :], Y[t, :])
        coeffs[t] = coeff
        
    return coeffs

def componentwise_cosine(X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    assert X.shape == Y.shape, "Signals must have the same shape"
    
    # Normalize vectors along feature dimension
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    
    # Avoid division by zero
    valid = (X_norm > 0) & (Y_norm > 0)
    
    sims = np.zeros(X.shape[0])
    sims[valid] = np.sum(X[valid] * Y[valid], axis=1) / (X_norm[valid] * Y_norm[valid])
    
    return sims

def time_to_rel_idx(t, start, sr=50):
    return int((t - start) * sr)