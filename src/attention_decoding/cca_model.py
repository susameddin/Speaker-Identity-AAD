import os
import random
import json
import torch
import scipy.io
import numpy as np
import soundfile as sf
import h5py
import mat73
import librosa
import random

from scipy import io, stats, linalg, interpolate
import matplotlib.pyplot as plt
from IPython.display import Audio, display

from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

from naplib.features import auditory_spectrogram


def lag_generator_new(r, lags):
    """
    Generate time-lagged version of neural responses.
    """
    lags = list(range(lags[0], lags[1] + 1))
    out = np.zeros([r.shape[0], r.shape[1] * len(lags)])
    # Zero-pad at the end to allow positive shifts
    r = np.pad(r, ((0, len(lags)), (0, 0)), 'constant')

    r_lag_list = []
    for lag in lags:
        t1 = np.roll(r, lag, axis=0)
        # Zero out wrapped samples so we only keep valid time points
        if lag < 0:
            t1[lag - 1:, :] = 0
        else:
            t1[:lag, :] = 0
        r_lag_list.append(t1[:out.shape[0], :])

    out = np.concatenate(r_lag_list, axis=1)
    return out


def corrnum(a, b):
    """
    Simple Pearson correlation between two 1D arrays.
    """
    a_m = np.mean(a)
    b_m = np.mean(b)

    c = (np.mean((a - a_m) * (b - b_m))) / np.sqrt(
        np.mean((a - a_m) ** 2) * np.mean((b - b_m) ** 2)
    )
    return c


def decode_acc_cca_whole(cca_fit_att, resp_test, stim_att_test, stim_unatt_test):
    """
    CCA-based attention decoding for a single trial.
    Compares correlation for attended vs unattended stimulus projections.
    """
    X_c_att, Y_c_att = cca_fit_att.transform(resp_test.T, stim_att_test.T)
    X_c_unatt, Y_c_unatt = cca_fit_att.transform(resp_test.T, stim_unatt_test.T)

    c1 = stats.pearsonr(
        np.squeeze(X_c_att.flatten()),
        np.squeeze(Y_c_att.flatten())
    ).statistic
    c2 = stats.pearsonr(
        np.squeeze(X_c_unatt.flatten()),
        np.squeeze(Y_c_unatt.flatten())
    ).statistic

    acc = (c1 > c2)
    return acc


# Paths and basic params 
clip_root = '/engram/naplab/shared/spk_id_aad/Clips'
train_file = os.path.join(clip_root, 'labels_train.json')
valid_file = os.path.join(clip_root, 'labels_valid.json')
test_file = os.path.join(clip_root, 'labels_test.json')

fs = 100          # neural sampling rate (Hz)
lags_neuro = [-40, 0]
lags_stim = [-20, 0]
soundf = 16000    # audio sampling rate


# Load clip metadata
with open(train_file, 'r') as f:
    train_clips = json.load(f)

with open(test_file, 'r') as f:
    test_clips = json.load(f)


# Channel selection 
# Load list of bad channels selected by visual inspection and obtain good ones
with open(os.path.join(clip_root, 'Data/badchannels_lowerlim.txt')) as file:
    bad_elecs = [int(line.rstrip()) - 1 for line in file]
good_elecs = np.setdiff1d(np.arange(1311), np.array(bad_elecs))

subject_id = 1
subject_list = np.load(os.path.join(clip_root, 'Data/subject_list.npy'))
elec_list = np.intersect1d(np.where(subject_list == subject_id), good_elecs)


# Build training matrices
resp_train = []
stim_att_train = []
stim_unatt_train = []

for clip in train_clips:
    # Load neural data
    neuro_path = os.path.join(
        "{root}/Clips",
        "NeuralData",
        clip['BGPath'].split("/")[-1][:-4] + "_resp_lf.pt"
    )
    neuro_path = neuro_path[:-3] + "_norm.pt"
    neuro = torch.load(neuro_path.format(root=clip_root)).numpy()
    neuro = neuro[:, elec_list].T  # [channels, time]

    # Load attended / unattended audio
    conv1_path = clip['Conv1Path']
    conv2_path = clip['Conv2Path']
    stim_att = librosa.load(conv1_path.format(root=clip_root), sr=soundf)[0]
    stim_unatt = librosa.load(conv2_path.format(root=clip_root), sr=soundf)[0]

    # Compute auditory spectrograms (time–frequency)
    stim_att = auditory_spectrogram(stim_att, sfreq=soundf).T
    stim_unatt = auditory_spectrogram(stim_unatt, sfreq=soundf).T

    # Interpolate spectrograms to match neural time axis
    x = list(range(stim_att.shape[1]))
    fun = interpolate.interp1d(x, stim_att, axis=1, kind='nearest')
    xnew = np.linspace(0, stim_att.shape[1] - 1, neuro.shape[1])
    stim_att = fun(xnew)

    x = list(range(stim_unatt.shape[1]))
    fun = interpolate.interp1d(x, stim_unatt, axis=1, kind='nearest')
    xnew = np.linspace(0, stim_unatt.shape[1] - 1, neuro.shape[1])
    stim_unatt = fun(xnew)

    # Concatenate data with small zero gaps between clips
    resp_train.append(neuro)
    resp_train.append(np.zeros((neuro.shape[0], int(0.1 * fs))))

    stim_att_train.append(stim_att)
    stim_att_train.append(np.zeros((stim_att.shape[0], int(0.1 * fs))))

    stim_unatt_train.append(stim_unatt)
    stim_unatt_train.append(np.zeros((stim_unatt.shape[0], int(0.1 * fs))))

# Stack all training trials
resp_train = np.concatenate(resp_train, axis=1)
stim_att_train = np.concatenate(stim_att_train, axis=1)
stim_unatt_train = np.concatenate(stim_unatt_train, axis=1)


# PCA on stimulus features 
# Fit PCA jointly on attended + unattended spectrograms
pca = PCA(n_components=0.95, svd_solver='full')
pca.fit(np.concatenate((stim_att_train.T, stim_unatt_train.T), axis=0))
stim_att_train = pca.transform(stim_att_train.T).T
stim_unatt_train = pca.transform(stim_unatt_train.T).T


# Add lags and fit CCA
resp_train = lag_generator_new(resp_train.T, lags_neuro).T
stim_att_train = lag_generator_new(stim_att_train.T, lags_stim).T
stim_unatt_train = lag_generator_new(stim_unatt_train.T, lags_stim).T

cca_att = CCA(n_components=3)
cca_fit_att = cca_att.fit(resp_train.T, stim_att_train.T)


# Test loop (CCA decoding)
pred_avg = 0
count = 0

for clip in test_clips:
    # Load neural data
    neuro_path = os.path.join(
        "{root}/Clips",
        "NeuralData",
        clip['BGPath'].split("/")[-1][:-4] + "_resp_lf.pt"
    )
    neuro_path = neuro_path[:-3] + "_norm.pt"
    neuro = torch.load(neuro_path.format(root=clip_root)).numpy()
    neuro = neuro[:, elec_list].T

    # Load attended / unattended audio
    conv1_path = clip['Conv1Path']
    conv2_path = clip['Conv2Path']

    stim_att = librosa.load(conv1_path.format(root=clip_root), sr=soundf)[0]
    stim_unatt = librosa.load(conv2_path.format(root=clip_root), sr=soundf)[0]

    stim_att = auditory_spectrogram(stim_att, sfreq=soundf).T
    stim_unatt = auditory_spectrogram(stim_unatt, sfreq=soundf).T

    # Interpolate to neural time
    x = list(range(stim_att.shape[1]))
    fun = interpolate.interp1d(x, stim_att, axis=1, kind='nearest')
    xnew = np.linspace(0, stim_att.shape[1] - 1, neuro.shape[1])
    stim_att = fun(xnew)

    x = list(range(stim_unatt.shape[1]))
    fun = interpolate.interp1d(x, stim_unatt, axis=1, kind='nearest')
    xnew = np.linspace(0, stim_unatt.shape[1] - 1, neuro.shape[1])
    stim_unatt = fun(xnew)

    resp_test = neuro
    stim_att_test = stim_att
    stim_unatt_test = stim_unatt

    # Apply same PCA + lagging as train
    stim_att_test = pca.transform(stim_att_test.T).T
    stim_unatt_test = pca.transform(stim_unatt_test.T).T

    resp_test = lag_generator_new(resp_test.T, lags_neuro).T
    stim_att_test = lag_generator_new(stim_att_test.T, lags_stim).T
    stim_unatt_test = lag_generator_new(stim_unatt_test.T, lags_stim).T

    # One-trial CCA-based decision
    pred = decode_acc_cca_whole(cca_fit_att, resp_test, stim_att_test, stim_unatt_test)
    pred_avg += pred

    clip['PredCCA'] = pred
    count += 1

pred_avg /= len(test_clips)
print(f"Prediction Average: {pred_avg}")

# Save predictions to JSON
test_pred_file = os.path.join(clip_root, 'labels_test_preds.json')

with open(test_pred_file, 'w') as f:
    json.dump(test_clips, f, indent=4)