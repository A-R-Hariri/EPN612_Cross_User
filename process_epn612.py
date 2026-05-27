import gc
from os.path import join
import numpy as np 
from numpy.lib.stride_tricks import sliding_window_view
import copy

from utils import *


def subsample_rest(windows, labels, subjects, keep_every_k=4):
    """
    Stride-subsample rest windows only.
    Every other class untouched.
    
    windows  : (N, 8, 40)
    labels   : (N,)
    subjects : (N,)
    """
    rest_idx   = np.where(labels == 0)[0]
    active_idx = np.where(labels != 0)[0]
    
    kept_rest  = rest_idx[::keep_every_k]
    keep_all   = np.sort(np.concatenate([kept_rest, active_idx]))
    
    print(f"Rest:   {len(rest_idx):,} → {len(kept_rest):,}")
    print(f"Active: {len(active_idx):,} → unchanged")
    
    return windows[keep_all], labels[keep_all], subjects[keep_all]

#======== DATA ========
os.makedirs(PICKLE_PATH, exist_ok=True)
from EPN612 import EMGEPN612
dataset = EMGEPN612()
data = dataset.prepare_data(split=True, segment=True, relabel_seg=None)


# ======== RAW ========
train_data = data['Train']
train_data = train_data.isolate_data("classes", [0, 1, 2, 3, 4], fast=True)

test_data = data['Test']
test_data = test_data.isolate_data("classes", [0, 1, 2, 3, 4], fast=True)
val_data = test_data.isolate_data("subjects", list(range(306, VAL_CUTOFF)), fast=True)
test_data = test_data.isolate_data("subjects", list(range(VAL_CUTOFF, 612)), fast=True)

np.save(join(PICKLE_PATH, 'train_data_raw'), train_data)
np.save(join(PICKLE_PATH, 'val_data_raw'), val_data)
np.save(join(PICKLE_PATH, 'test_data_raw'), test_data)

train_windows, train_meta = train_data.parse_windows(SEQ, INC)
train_windows, train_meta['classes'], train_meta['subjects'] = subsample_rest(
                    train_windows, train_meta['classes'], train_meta['subjects']
                    )
np.save(join(PICKLE_PATH, 'train_windows_raw'), train_windows.astype(DTYPE))
np.save(join(PICKLE_PATH, 'train_meta_raw'), train_meta)
del train_windows
gc.collect()

val_windows, val_meta = val_data.parse_windows(SEQ, INC)
val_windows, val_meta['classes'], val_meta['subjects'] = subsample_rest(
                    val_windows, val_meta['classes'], val_meta['subjects']
                    )
np.save(join(PICKLE_PATH, 'val_windows_raw'), val_windows.astype(DTYPE))
np.save(join(PICKLE_PATH, 'val_meta_raw'), val_meta)
del val_windows
gc.collect()

test_windows, test_meta = test_data.parse_windows(SEQ, INC)
test_windows, test_meta['classes'], test_meta['subjects'] = subsample_rest(
                    test_windows, test_meta['classes'], test_meta['subjects']
                    )
np.save(join(PICKLE_PATH, 'test_windows_raw'), test_windows.astype(DTYPE))
np.save(join(PICKLE_PATH, 'test_meta_raw'), test_meta)
del test_windows
gc.collect()

# ======== SEGMENTED ========
train_data = np.load(join(PICKLE_PATH, 'train_data_raw.npy'), allow_pickle=True).item()
val_data = np.load(join(PICKLE_PATH, 'val_data_raw.npy'), allow_pickle=True).item()
test_data = np.load(join(PICKLE_PATH, 'test_data_raw.npy'), allow_pickle=True).item()
train_data_segmented = copy.deepcopy(train_data)
val_data_segmented = copy.deepcopy(val_data)
test_data_segmented = copy.deepcopy(test_data)

del train_data
del val_data
del test_data
gc.collect()

def tkeo(x):
    return x[1:-1]**2 - x[:-2] * x[2:]

def extract_active_segment(emg_data, window_size=5, threshold=0.25, 
                        n_samples=SEQ + INC, method='energy'):
    segmented_data = []
    segmented_classes = []
    segmented_reps = []
    segmented_subjects = []
    segmented_sb = []
    segmented_se = []

    total_original = 0
    total_kept = 0

    for i in range(len(emg_data.data)):
        data_i = np.asarray(emg_data.data[i])
        class_i = np.asarray(emg_data.classes[i])
        rep_i = np.asarray(emg_data.reps[i])
        subj_i = np.asarray(emg_data.subjects[i])

        t, ch = data_i.shape
        assert ch == 8, f"Expected 8 channels, got {ch} at index {i}"
        total_original += t

        for meta_arr, name in zip([class_i, rep_i, subj_i], ['classes', 'reps', 'subjects']):
            if meta_arr.shape != (t, 1):
                raise ValueError(f"{name}[{i}] must have shape (t, 1), got {meta_arr.shape}")
            if not np.all(meta_arr == meta_arr[0]):
                raise ValueError(f"{name}[{i}] is not constant across time")

        current_class = class_i[0, 0]

        if method == 'tkeo':
            best_val = -np.inf
            signal = None
            for ch_idx in range(ch):
                x = data_i[:, ch_idx]
                if len(x) < 3:
                    continue
                e = tkeo(x)
                smoothed = np.convolve(e, np.ones(window_size) / window_size, mode='same')
                max_energy = np.max(smoothed)
                if max_energy > best_val:
                    best_val = max_energy
                    signal = smoothed
            if signal is None:
                signal = np.zeros(t - 2)
                signal = np.pad(signal, (1, 1), mode='constant')

        elif method == 'energy':
            ch_candidates = list(range(ch))

            channel_energies = [np.sum(data_i[:, idx] ** 2) for idx in ch_candidates]
            main_ch_idx = ch_candidates[np.argmax(channel_energies)]
            signal = data_i[:, main_ch_idx] ** 2
            signal = np.convolve(signal, np.ones(window_size) / window_size, mode='same')

        elif method == 'variance':
            if current_class == 3:
                ch_candidates = [3, 4]
            elif current_class == 4:
                ch_candidates = [0, 7]
            else:
                ch_candidates = list(range(ch))

            mavs = [np.var(np.abs(data_i[:, idx])) for idx in ch_candidates]
            main_ch_idx = ch_candidates[np.argmax(mavs)]
            signal = np.abs(data_i[:, main_ch_idx])
            signal = np.convolve(signal, np.ones(window_size) / window_size, mode='same')

        signal_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
        active_indices = np.where(signal_norm > threshold)[0]

        if len(active_indices) > 1:
            start_idx, end_idx = active_indices[0], active_indices[-1] + 1
        else:
            start_idx, end_idx = 0, t

        if (end_idx - start_idx) <= n_samples or current_class == 0 or len(active_indices) == 0:
            start_idx, end_idx = 0, t

        total_kept += end_idx - start_idx
        segmented_data.append(data_i[start_idx:end_idx])
        segmented_classes.append(class_i[start_idx:end_idx])
        segmented_reps.append(rep_i[start_idx:end_idx])
        segmented_subjects.append(subj_i[start_idx:end_idx])
        segmented_sb.append(np.array([[start_idx] for _ in range(end_idx - start_idx)]))
        segmented_se.append(np.array([[end_idx] for _ in range(end_idx - start_idx)]))

    percent_removed = 100 * (total_original - total_kept) / total_original
    print(f"Total data removed by segmenting: {percent_removed:.2f}%")

    return (
        segmented_data,
        segmented_classes,
        segmented_reps,
        segmented_subjects,
        segmented_sb,
        segmented_se
    )

train_data_segmented.extra_attributes.append("sb")
train_data_segmented.extra_attributes.append("se")
train_data_segmented.data, train_data_segmented.classes, train_data_segmented.reps, \
    train_data_segmented.subjects, train_data_segmented.sb, train_data_segmented.se = extract_active_segment(train_data_segmented)
train_data_segmented.base_class = train_data_segmented.classes

val_data_segmented.extra_attributes.append("sb")
val_data_segmented.extra_attributes.append("se")
val_data_segmented.data, val_data_segmented.classes, val_data_segmented.reps, \
    val_data_segmented.subjects, val_data_segmented.sb, val_data_segmented.se = extract_active_segment(val_data_segmented)
val_data_segmented.base_class = val_data_segmented.classes

test_data_segmented.extra_attributes.append("sb")
test_data_segmented.extra_attributes.append("se")
test_data_segmented.data, test_data_segmented.classes, test_data_segmented.reps, \
    test_data_segmented.subjects, test_data_segmented.sb, test_data_segmented.se = extract_active_segment(test_data_segmented)
test_data_segmented.base_class = test_data_segmented.classes

train_windows_segmented, train_meta_segmented = train_data_segmented.parse_windows(SEQ, INC)
val_windows_segmented, val_meta_segmented = val_data_segmented.parse_windows(SEQ, INC)
test_windows_segmented, test_meta_segmented = test_data_segmented.parse_windows(SEQ, INC)

np.save(join(PICKLE_PATH, 'train_data_segmented'), train_data_segmented)
np.save(join(PICKLE_PATH, 'val_data_segmented'), val_data_segmented)
np.save(join(PICKLE_PATH, 'test_data_segmented'), test_data_segmented)

np.save(join(PICKLE_PATH, 'train_windows_segmented'), train_windows_segmented.astype(DTYPE))
np.save(join(PICKLE_PATH, 'train_meta_segmented'), train_meta_segmented)
np.save(join(PICKLE_PATH, 'val_windows_segmented'), val_windows_segmented.astype(DTYPE))
np.save(join(PICKLE_PATH, 'val_meta_segmented'), val_meta_segmented)
np.save(join(PICKLE_PATH, 'test_windows_segmented'), test_windows_segmented.astype(DTYPE))
np.save(join(PICKLE_PATH, 'test_meta_segmented'), test_meta_segmented)

del train_data_segmented
del val_data_segmented
del test_data_segmented
del train_windows_segmented
del train_meta_segmented
del val_windows_segmented
del val_meta_segmented
del test_windows_segmented
del test_meta_segmented
gc.collect()

# ======== RELABELED ========
train_data = np.load(join(PICKLE_PATH, 'train_data_raw.npy'), allow_pickle=True).item()
val_data = np.load(join(PICKLE_PATH, 'val_data_raw.npy'), allow_pickle=True).item()
test_data = np.load(join(PICKLE_PATH, 'test_data_raw.npy'), allow_pickle=True).item()

train_s = np.load(join(PICKLE_PATH, 'train_data_segmented.npy'), allow_pickle=True).item()
val_s = np.load(join(PICKLE_PATH, 'val_data_segmented.npy'), allow_pickle=True).item()
test_s = np.load(join(PICKLE_PATH, 'test_data_segmented.npy'), allow_pickle=True).item()

for i in range(len(train_data.data)):
    if train_data.classes[i][0].item() == 0:
        continue

    sb, se = train_s.sb[i][0][0], train_s.se[i][0][0]

    if se - sb > SEQ + INC:
        train_data.data[i] = train_data.data[i][sb:se]
        train_data.reps[i] = train_data.reps[i][sb:se]
        train_data.classes[i] = train_data.classes[i][sb:se]
        train_data.subjects[i] = train_data.subjects[i][sb:se]
        train_data.base_class[i] = train_data.base_class[i][sb:se]

    if len(train_data.data[i][:sb]) > SEQ + INC:
        train_data.data.append(train_data.data[i][:sb])
        train_data.reps.append(train_data.reps[i][:sb])
        train_data.classes.append(np.zeros_like(train_data.classes[i][:sb]))
        train_data.subjects.append(train_data.subjects[i][:sb])
        train_data.base_class.append(np.zeros_like(train_data.base_class[i][:sb]))

    if len(train_data.data[i][se:]) > SEQ + INC:
        train_data.data.append(train_data.data[i][se:])
        train_data.reps.append(train_data.reps[i][se:])
        train_data.classes.append(np.zeros_like(train_data.classes[i][se:]))
        train_data.subjects.append(train_data.subjects[i][se:])
        train_data.base_class.append(np.zeros_like(train_data.base_class[i][se:]))

for i in range(len(val_data.data)):
    if val_data.classes[i][0].item() == 0:
        continue

    sb, se = val_s.sb[i][0][0], val_s.se[i][0][0]

    if se - sb > SEQ + INC:
        val_data.data[i] = val_data.data[i][sb:se]
        val_data.reps[i] = val_data.reps[i][sb:se]
        val_data.classes[i] = val_data.classes[i][sb:se]
        val_data.subjects[i] = val_data.subjects[i][sb:se]
        val_data.base_class[i] = val_data.base_class[i][sb:se]

    if len(val_data.data[i][:sb]) > SEQ + INC:
        val_data.data.append(val_data.data[i][:sb])
        val_data.reps.append(val_data.reps[i][:sb])
        val_data.classes.append(np.zeros_like(val_data.classes[i][:sb]))
        val_data.subjects.append(val_data.subjects[i][:sb])
        val_data.base_class.append(np.zeros_like(val_data.base_class[i][:sb]))

    if len(val_data.data[i][se:]) > SEQ + INC:
        val_data.data.append(val_data.data[i][se:])
        val_data.reps.append(val_data.reps[i][se:])
        val_data.classes.append(np.zeros_like(val_data.classes[i][se:]))
        val_data.subjects.append(val_data.subjects[i][se:])
        val_data.base_class.append(np.zeros_like(val_data.base_class[i][se:]))

for i in range(len(test_data.data)):
    if test_data.classes[i][0].item() == 0:
        continue

    sb, se = test_s.sb[i][0][0], test_s.se[i][0][0]

    if se - sb > SEQ + INC:
        test_data.data[i] = test_data.data[i][sb:se]
        test_data.reps[i] = test_data.reps[i][sb:se]
        test_data.classes[i] = test_data.classes[i][sb:se]
        test_data.subjects[i] = test_data.subjects[i][sb:se]
        test_data.base_class[i] = test_data.base_class[i][sb:se]

    if len(test_data.data[i][:sb]) > SEQ + INC:
        test_data.data.append(test_data.data[i][:sb])
        test_data.reps.append(test_data.reps[i][:sb])
        test_data.classes.append(np.zeros_like(test_data.classes[i][:sb]))
        test_data.subjects.append(test_data.subjects[i][:sb])
        test_data.base_class.append(np.zeros_like(test_data.base_class[i][:sb]))

    if len(test_data.data[i][se:]) > SEQ + INC:
        test_data.data.append(test_data.data[i][se:])
        test_data.reps.append(test_data.reps[i][se:])
        test_data.classes.append(np.zeros_like(test_data.classes[i][se:]))
        test_data.subjects.append(test_data.subjects[i][se:])
        test_data.base_class.append(np.zeros_like(test_data.base_class[i][se:]))

X, y = train_data.parse_windows(SEQ, INC)
X_v, y_v = val_data.parse_windows(SEQ, INC)
X_t, y_t = test_data.parse_windows(SEQ, INC)

np.save(join(PICKLE_PATH, 'train_windows_relabeled'), X.astype(DTYPE))
np.save(join(PICKLE_PATH, 'train_meta_relabeled'), y)
np.save(join(PICKLE_PATH, 'val_windows_relabeled'), X_v.astype(DTYPE))
np.save(join(PICKLE_PATH, 'val_meta_relabeled'), y_v)
np.save(join(PICKLE_PATH, 'test_windows_relabeled'), X_t.astype(DTYPE))
np.save(join(PICKLE_PATH, 'test_meta_relabeled'), y_t)

np.save(join(PICKLE_PATH, 'train_data_relabeled'), train_data)
np.save(join(PICKLE_PATH, 'val_data_relabeled'), val_data)
np.save(join(PICKLE_PATH, 'test_data_relabeled'), test_data)

del train_data
del val_data
del test_data
del X, y
del X_v, y_v
del X_t, y_t
gc.collect()

# ======== STANDARD ========
def detect_segments(x, thresh, win_len=SEQ):
    x = np.abs(x)      
    T, C = x.shape
    if T < win_len:
        return -1, -1

    windows = sliding_window_view(x, window_shape=win_len, axis=0) 
    vals = windows.mean(axis=(1,2))
    mask = vals > thresh
    if mask.any():
        idx = np.where(mask)[0] 
        first_idx = idx[0]
        last_idx  = idx[-1] + win_len - 1
    else:
        first_idx = 0
        last_idx  = T
    if (last_idx - first_idx) <=  SEQ + INC:
        first_idx = 0
        last_idx  = T
    return first_idx, last_idx

train_data = np.load(join(PICKLE_PATH, 'train_data_raw.npy'), allow_pickle=True).item()
val_data = np.load(join(PICKLE_PATH, 'val_data_raw.npy'), allow_pickle=True).item()
test_data = np.load(join(PICKLE_PATH, 'test_data_raw.npy'), allow_pickle=True).item()

train_data_standard = copy.deepcopy(train_data)
val_data_standard = copy.deepcopy(val_data)
test_data_standard = copy.deepcopy(test_data)

train_windows = np.load(join(PICKLE_PATH, 'train_windows_raw.npy'))
train_meta = np.load(join(PICKLE_PATH, 'train_meta_raw.npy'), allow_pickle=True).item()

x_rst = train_windows[train_meta['classes'] == 0]
feat = np.abs(x_rst).mean(axis=(-1, -2))
mu = feat.mean()
sigma = feat.std()
thresh = mu + 3 * sigma

total = []

for i in range(len(train_data.data)):
    if train_data.classes[i][0].item() == 0:
        total.append(1.0)
        continue
    sb, se = detect_segments(train_data.data[i], mu + 3 * sigma)
    train_data_standard.data[i] = train_data.data[i][sb:se]
    train_data_standard.reps[i] = train_data.reps[i][sb:se]
    train_data_standard.classes[i] = train_data.classes[i][sb:se]
    train_data_standard.subjects[i] = train_data.subjects[i][sb:se]
    train_data_standard.reps[i] = train_data.reps[i][sb:se]
    train_data_standard.base_class[i] = train_data.base_class[i][sb:se]
    total.append(len(train_data.data[i][sb:se]) / len(train_data.data[i]))

for i in range(len(val_data.data)):
    if val_data.classes[i][0].item() == 0:
        total.append(1.0)
        continue
    sb, se = detect_segments(val_data.data[i], mu + 3 * sigma)
    val_data_standard.data[i] = val_data.data[i][sb:se]
    val_data_standard.reps[i] = val_data.reps[i][sb:se]
    val_data_standard.classes[i] = val_data.classes[i][sb:se]
    val_data_standard.subjects[i] = val_data.subjects[i][sb:se]
    val_data_standard.reps[i] = val_data.reps[i][sb:se]
    val_data_standard.base_class[i] = val_data.base_class[i][sb:se]
    total.append(len(val_data.data[i][sb:se]) / len(val_data.data[i]))

for i in range(len(test_data.data)):
    if test_data.classes[i][0].item() == 0:
        total.append(1.0)
        continue
    sb, se = detect_segments(test_data.data[i], mu + 3 * sigma)
    test_data_standard.data[i] = test_data.data[i][sb:se]
    test_data_standard.reps[i] = test_data.reps[i][sb:se]
    test_data_standard.classes[i] = test_data.classes[i][sb:se]
    test_data_standard.subjects[i] = test_data.subjects[i][sb:se]
    test_data_standard.reps[i] = test_data.reps[i][sb:se]
    test_data_standard.base_class[i] = test_data.base_class[i][sb:se]
    total.append(len(test_data.data[i][sb:se]) / len(test_data.data[i]))

X, y = train_data_standard.parse_windows(SEQ, INC)
X_v, y_v = val_data_standard.parse_windows(SEQ, INC)
X_t, y_t = test_data_standard.parse_windows(SEQ, INC)

np.save(join(PICKLE_PATH, 'train_windows_standard'), X.astype(DTYPE))
np.save(join(PICKLE_PATH, 'train_meta_standard'), y)
np.save(join(PICKLE_PATH, 'val_windows_standard'), X_v.astype(DTYPE))
np.save(join(PICKLE_PATH, 'val_meta_standard'), y_v)
np.save(join(PICKLE_PATH, 'test_windows_standard'), X_t.astype(DTYPE))
np.save(join(PICKLE_PATH, 'test_meta_standard'), y_t)

np.save(join(PICKLE_PATH, 'train_data_standard'), train_data_standard)
np.save(join(PICKLE_PATH, 'val_data_standard'), val_data_standard)
np.save(join(PICKLE_PATH, 'test_data_standard'), test_data_standard)

print(np.mean(total))


print('DONE.')