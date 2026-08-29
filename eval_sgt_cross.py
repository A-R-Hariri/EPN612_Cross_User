# Test online study participants' SGT data with EPN-trained cross-user models.

import warnings, sys, os
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"

import numpy as np
import torch
import libemg
from os.path import join, exists
from sklearn.metrics import f1_score, confusion_matrix

from utils import (SEQ, INC, CH, CLASSES, DEVICE, CHECKPOINT_PATH,
                   TAU, RunningNorm)
from models import MHCNN


USER_SGT_ROOT = "user_sgt"
N_USERS       = 16
VAL_REP       = [5]           # rep 5 is the held-out validation rep

pop_mean = np.array([-0.70885944, -0.74997824, -0.47742087, -0.73471236,
                     -0.99069226, -0.9039961,  -0.8920331,  -0.78142345],
                    dtype=np.float32)
pop_std  = np.array([21.533228, 21.636055, 28.874157, 30.270008,
                     17.713427, 13.582925, 19.829926, 21.28387],
                    dtype=np.float32)

mapping = {0: 1, 1: 4, 2: 0, 3: 3, 4: 2}
def remap_labels(labels, mapping=mapping):
    return np.array([mapping[x] for x in labels])

MODELS = [
    ("cross_mhcnn_raw_base",       "model_state_dict", False, False),
    ("cross_mhcnn_raw_1va",        "model_state_dict", False, False),
    ("cross_mhcnn_raw_base-rn",    "model_state_dict", True,  False),
    ("cross_mhcnn_raw_rest",       "model_state_dict", False, False),
    ("cross_mhcnn_raw_trp",        "model_state_dict", False, False),
    ("cross_mhcnn_segmented_base", "model_state_dict", False, True),
]


# ======== segmentation ========

def tkeo(x):
    return x[1:-1]**2 - x[:-2] * x[2:]
 
 
def extract_active_segment(emg_data, window_size=5, threshold=0.25,
                            n_samples=SEQ + INC, method='energy'):
    seg_data, seg_classes, seg_reps, seg_subjects = [], [], [], []
    seg_sb, seg_se = [], []
 
    total_original = total_kept = 0
 
    for i in range(len(emg_data.data)):
        data_i  = np.asarray(emg_data.data[i])
        class_i = np.asarray(emg_data.classes[i])
        rep_i   = np.asarray(emg_data.reps[i])
        # subj_i  = np.asarray(emg_data.subjects[i])
 
        t, ch = data_i.shape
        total_original += t
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
                signal = np.zeros(t)
 
        elif method == 'energy':
            channel_energies = [np.sum(data_i[:, idx] ** 2) for idx in range(ch)]
            main_ch_idx = int(np.argmax(channel_energies))
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
            main_ch_idx = ch_candidates[int(np.argmax(mavs))]
            signal = np.abs(data_i[:, main_ch_idx])
            signal = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
 
        else:
            raise ValueError(f"Unknown method: {method}")
 
        sig_min, sig_max = signal.min(), signal.max()
        signal_norm = (signal - sig_min) / (sig_max - sig_min + 1e-8)
        active_indices = np.where(signal_norm > threshold)[0]
 
        if len(active_indices) > 1:
            start_idx = active_indices[0]
            end_idx   = active_indices[-1] + 1
        else:
            start_idx, end_idx = 0, t
 
        # Fall back to full trial if segment is too short, class is rest, or no
        # active region was found.
        if (end_idx - start_idx) <= n_samples or current_class == 0 or len(active_indices) == 0:
            start_idx, end_idx = 0, t
 
        total_kept += end_idx - start_idx
        seg_data.append(data_i[start_idx:end_idx])
        seg_classes.append(class_i[start_idx:end_idx])
        seg_reps.append(rep_i[start_idx:end_idx])
        # seg_subjects.append(subj_i[start_idx:end_idx])
        seg_sb.append(np.array([[start_idx]] * (end_idx - start_idx)))
        seg_se.append(np.array([[end_idx]]   * (end_idx - start_idx)))
 
    pct_removed = 100 * (total_original - total_kept) / max(1, total_original)
    print(f"    Segmentation: {pct_removed:.1f}% of samples removed")
 
    return seg_data, seg_classes, seg_reps, seg_subjects, seg_sb, seg_se
 
 
def apply_segmentation(odh):
    odh.extra_attributes = getattr(odh, 'extra_attributes', [])
    odh.extra_attributes.append("sb")
    odh.extra_attributes.append("se")

    (odh.data, odh.classes, odh.reps,
     _, odh.sb, odh.se) = extract_active_segment(odh)
    odh.base_class = odh.classes
    return odh


# ======== data loading ========

def get_odh(user_id):
    folder = join(USER_SGT_ROOT, str(user_id))
    filters = [
        libemg.data_handler.RegexFilter(
            left_bound="C_", right_bound="_R",
            values=["0","1","2","3","4"], description="classes"),
        libemg.data_handler.RegexFilter(
            left_bound="R_", right_bound="_emg.csv",
            values=[str(r) for r in range(6)], description="reps"),
    ]
    odh_all = libemg.data_handler.OfflineDataHandler()
    odh_all.get_data(folder_location=folder, regex_filters=filters, delimiter=",")
    return odh_all


def windows_from_odh(odh_all, reps, segment=False):
    odh = odh_all.isolate_data("reps", reps, fast=True)
    if segment:
        odh = apply_segmentation(odh)
    windows, meta = odh.parse_windows(SEQ, INC)
    meta["classes"] = remap_labels(meta["classes"])
    return windows, meta["classes"]


# ======== model loading ========

def load_model(name, ckpt_key):
    model = MHCNN().to(DEVICE)
    ckpt_path = join(CHECKPOINT_PATH, name, f"{name}.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt if ckpt_key is None else ckpt[ckpt_key])
    model.eval()
    return model


# ======== inference ========

@torch.no_grad()
def infer(model, windows, labels, use_rn=False):
    x = torch.from_numpy(windows.astype(np.float32))

    rn = None
    if use_rn:
        rn = RunningNorm(CH, tau=TAU, init_mean=pop_mean, init_std=pop_std)
        rn = rn.to(DEVICE).eval()

    preds = []
    for i in range(len(x)):
        xb = x[i:i+1].to(DEVICE)
        if rn is not None:
            xb = rn(xb)
        preds.append(model(xb).argmax(1).item())

    preds  = np.array(preds)
    labels = labels.astype(int)

    acc     = (preds == labels).mean()
    act_m   = labels != 0
    act_acc = (preds[act_m] == labels[act_m]).mean() if act_m.any() else np.nan
    cm      = confusion_matrix(labels, preds, labels=np.arange(CLASSES))
    with np.errstate(divide="ignore", invalid="ignore"):
        per_cls = np.diag(cm) / cm.sum(axis=1)
    bal_acc = np.nanmean(per_cls)
    f1      = f1_score(labels, preds, average="macro", zero_division=0)

    return acc, act_acc, bal_acc, f1


# ======== main ========

def main():
    user_ids = []
    for uid in range(1, N_USERS + 1):
        folder = join(USER_SGT_ROOT, str(uid))
        if exists(folder) and os.listdir(folder):
            user_ids.append(uid)

    if not user_ids:
        print(f"No user data found under {USER_SGT_ROOT}/.")
        return

    print(f"Found {len(user_ids)} users: {user_ids}")
    print(f"Val rep: {VAL_REP}\n")

    print("Loading data handlers ...")
    odh_cache = {}
    for uid in user_ids:
        try:
            odh_cache[uid] = get_odh(uid)
            print(f"  User {uid:2d}: loaded")
        except Exception as e:
            print(f"  User {uid:2d}: FAILED — {e}")

    print("\nExtracting raw val windows ...")
    raw_data = {}
    for uid, odh_all in odh_cache.items():
        try:
            w, c = windows_from_odh(odh_all, VAL_REP, segment=False)
            raw_data[uid] = (w, c)
            print(f"  User {uid:2d}: {w.shape[0]} windows")
        except Exception as e:
            print(f"  User {uid:2d}: {e}")

    print("\nExtracting segmented val windows ...")
    seg_data = {}
    for uid, odh_all in odh_cache.items():
        # try:
        w, c = windows_from_odh(odh_all, VAL_REP, segment=True)
        seg_data[uid] = (w, c)
        print(f"  User {uid:2d}: {w.shape[0]} windows")
        # except Exception as e:
        #     print(f"  User {uid:2d}: {e}")

    sep = "=" * 72
    results = {}

    for name, ckpt_key, use_rn, use_seg in MODELS:
        ckpt_path = join(CHECKPOINT_PATH, name, f"{name}.pt")
        if not exists(ckpt_path):
            print(f"\n[SKIP] {name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{sep}")
        tags = []
        if use_rn:  tags.append("RunningNorm")
        if use_seg: tags.append("segmented")
        print(f"Model: {name}" + (f"  [{', '.join(tags)}]" if tags else ""))
        print(sep)

        try:
            model = load_model(name, ckpt_key)
        except Exception as e:
            print(f"  Failed to load checkpoint: {e}")
            continue

        data_dict = seg_data if use_seg else raw_data
        rows = []
        for uid in user_ids:
            if uid not in data_dict:
                continue
            w, c = data_dict[uid]
            try:
                acc, act_acc, bal_acc, f1 = infer(model, w, c, use_rn=use_rn)
                rows.append((acc, act_acc, bal_acc, f1))
                print(f"  User {uid:2d}  acc={acc*100:5.1f}%  "
                      f"act={act_acc*100:5.1f}%  "
                      f"bal={bal_acc*100:5.1f}%  "
                      f"f1={f1*100:5.1f}%")
            except Exception as e:
                print(f"  User {uid:2d}: inference failed — {e}")

        if rows:
            arr = np.array(rows) * 100
            m = arr.mean(axis=0)
            s = arr.std(axis=0, ddof=1) if len(rows) > 1 else np.zeros(4)
            print(f"\n  MEAN +- STD  "
                  f"acc={m[0]:.2f}±{s[0]:.2f}  "
                  f"act={m[1]:.2f}±{s[1]:.2f}  "
                  f"bal={m[2]:.2f}±{s[2]:.2f}  "
                  f"f1={m[3]:.2f}±{s[3]:.2f}")
            results[name] = rows

    # ======== summary table ========
    if results:
        print(f"\n{sep}")
        print("SUMMARY  (all values in %,  mean +- std across users)")
        print(f"{'Model':<35} {'Acc':>12} {'ActAcc':>14} {'BalAcc':>14} {'F1':>12}")
        print("-" * 72)
        for name, rows in results.items():
            arr = np.array(rows) * 100
            m = arr.mean(axis=0)
            s = arr.std(axis=0, ddof=1) if len(rows) > 1 else np.zeros(4)
            print(f"{name:<35} "
                  f"{m[0]:5.1f}±{s[0]:4.1f}  "
                  f"{m[1]:5.1f}±{s[1]:4.1f}  "
                  f"{m[2]:5.1f}±{s[2]:.1f}  "
                  f"{m[3]:5.1f}±{s[3]:.1f}")
        print(sep)


if __name__ == "__main__":
    main()