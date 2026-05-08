import warnings, sys, os, gc
from os.path import join
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"

import torch; print(torch.cuda.is_available())

import libemg
from libemg.datasets import get_dataset_list
from libemg.feature_extractor import FeatureExtractor

import numpy as np, pandas as pd
import random, copy, time
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from utils import *
from models import *


SEED = 13; random.seed(SEED); np.random.seed(SEED)
GENERATOR = torch.manual_seed(SEED)
MMAP_MODE = 'r'; SAVE_CHKP = True

EPOCHS = 200; BATCH_SIZE = 4096; DROPOUT = 0.2; PATIENCE = 40
LR_FACTOR = 0.5; LR_PATIENCE = 8; LR_INIT = 1e-4; LR_MIN = 1e-6

N_SUBJECTS = 306; MARGIN = 0.5
ALPHA_START = 0.0; ALPHA_END = 0.25; WARMUP = 25


# ======== LOAD DATA ========
train_windows = np.load(join(PICKLE_PATH, 'train_windows.npy'), mmap_mode=MMAP_MODE)
train_meta = np.load(join(PICKLE_PATH, 'train_meta.npy'), allow_pickle=True).item()
val_windows = np.load(join(PICKLE_PATH, 'val_windows.npy'), mmap_mode=MMAP_MODE)
val_meta = np.load(join(PICKLE_PATH, 'val_meta.npy'), allow_pickle=True).item()
test_windows = np.load(join(PICKLE_PATH, 'test_windows.npy'), mmap_mode=MMAP_MODE)
test_meta = np.load(join(PICKLE_PATH, 'test_meta.npy'), allow_pickle=True).item()

train_windows_segmented = np.load(join(PICKLE_PATH, 'train_windows_segmented.npy'), mmap_mode=MMAP_MODE)
train_meta_segmented = np.load(join(PICKLE_PATH, 'train_meta_segmented.npy'), allow_pickle=True).item()
val_windows_segmented = np.load(join(PICKLE_PATH, 'val_windows_segmented.npy'), mmap_mode=MMAP_MODE)
val_meta_segmented = np.load(join(PICKLE_PATH, 'val_meta_segmented.npy'), allow_pickle=True).item()
test_windows_segmented = np.load(join(PICKLE_PATH, 'test_windows_segmented.npy'), mmap_mode=MMAP_MODE)
test_meta_segmented = np.load(join(PICKLE_PATH, 'test_meta_segmented.npy'), allow_pickle=True).item()

train_windows_relabeled = np.load(join(PICKLE_PATH, 'train_windows_relabeled.npy'), mmap_mode=MMAP_MODE)
train_meta_relabeled = np.load(join(PICKLE_PATH, 'train_meta_relabeled.npy'), allow_pickle=True).item()
val_windows_relabeled = np.load(join(PICKLE_PATH, 'val_windows_relabeled.npy'), mmap_mode=MMAP_MODE)
val_meta_relabeled = np.load(join(PICKLE_PATH, 'val_meta_relabeled.npy'), allow_pickle=True).item()
test_windows_relabeled = np.load(join(PICKLE_PATH, 'test_windows_relabeled.npy'), mmap_mode=MMAP_MODE)
test_meta_relabeled = np.load(join(PICKLE_PATH, 'test_meta_relabeled.npy'), allow_pickle=True).item()

train_windows_standard = np.load(join(PICKLE_PATH, 'train_windows_standard.npy'), mmap_mode=MMAP_MODE)
train_meta_standard = np.load(join(PICKLE_PATH, 'train_meta_standard.npy'), allow_pickle=True).item()
val_windows_standard = np.load(join(PICKLE_PATH, 'val_windows_standard.npy'), mmap_mode=MMAP_MODE)
val_meta_standard = np.load(join(PICKLE_PATH, 'val_meta_standard.npy'), allow_pickle=True).item()
test_windows_standard = np.load(join(PICKLE_PATH, 'test_windows_standard.npy'), mmap_mode=MMAP_MODE)
test_meta_standard = np.load(join(PICKLE_PATH, 'test_meta_standard.npy'), allow_pickle=True).item()

# ======== PIPELINE ========
test_loader = create_loader(test_windows, test_meta['classes'], 
                            batch=BATCH_SIZE, shuffle=False)
test_loader_segmented = create_loader(test_windows_segmented, 
                            test_meta_segmented['classes'], 
                            batch=BATCH_SIZE, shuffle=False)
test_loader_relabeled = create_loader(test_windows_relabeled, 
                            test_meta_relabeled['classes'],
                            batch=BATCH_SIZE, shuffle=False)
test_loader_standard = create_loader(test_windows_standard, 
                            test_meta_standard['classes'],
                            batch=BATCH_SIZE, shuffle=False)


DATA_TYPE = sys.argv[2] if len(sys.argv) > 2 else "all"
MODE = sys.argv[3] if len(sys.argv) > 3 else "train"


if DATA_TYPE == 'raw' or DATA_TYPE == 'all':
    train_loader = create_triplet_loader(
        train_windows, train_meta['classes'], train_meta['subjects'],
        batch=BATCH_SIZE, n_classes=CLASSES, n_subjects=N_SUBJECTS)
    val_loader_triplet = create_triplet_loader(
        val_windows, val_meta['classes'], val_meta['subjects'],
        batch=BATCH_SIZE, n_classes=CLASSES, n_subjects=26)

    NAME = f"cnn_raw_trp"
    model = CNN()
    print(model, f"\nParameters count: {count_params(model):,}")

    if MODE == "train":
        train_triplet(model=model, name=NAME, 
            train_loader=train_loader, val_loader=val_loader_triplet, 
            criterion_ce=nn.CrossEntropyLoss(weight=None),
            criterion_tri=TripletLoss(margin= MARGIN, batch_hard=True), 
            save_chkp=SAVE_CHKP, epochs=EPOCHS, lr=LR_INIT, min_lr=LR_MIN, 
            lr_factor=LR_FACTOR, lr_patience=LR_PATIENCE, patience=PATIENCE,
            alpha_start=ALPHA_START, alpha_end=ALPHA_END, warmup_epochs=WARMUP)
        torch.save(model.state_dict(), join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
    else:
        model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))

    eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader},
           metas={'raw': test_meta})


if DATA_TYPE == 'standard' or DATA_TYPE == 'all':
    train_loader_standard = create_triplet_loader(
        train_windows_standard, train_meta_standard['classes'], train_meta_standard['subjects'],
        batch=BATCH_SIZE, n_classes=CLASSES, n_subjects=N_SUBJECTS)
    val_loader_triplet_standard = create_triplet_loader(
        val_windows_standard, val_meta_standard['classes'], val_meta_standard['subjects'],
        batch=BATCH_SIZE, n_classes=CLASSES, n_subjects=26)

    NAME = f"cnn_standard_trp"
    model = CNN()
    print(model, f"\nParameters count: {count_params(model):,}")

    if MODE == "train":
        train_triplet(model=model, name=NAME, 
            train_loader=train_loader_standard, val_loader=val_loader_triplet_standard, 
            criterion_ce=nn.CrossEntropyLoss(weight=None),
            criterion_tri=TripletLoss(margin= MARGIN, batch_hard=True), 
            save_chkp=SAVE_CHKP, epochs=EPOCHS, lr=LR_INIT, min_lr=LR_MIN, 
            lr_factor=LR_FACTOR, lr_patience=LR_PATIENCE, patience=PATIENCE,
            alpha_start=ALPHA_START, alpha_end=ALPHA_END, warmup_epochs=WARMUP)
        torch.save(model.state_dict(), join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
    else:
        model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))

    eval_test(model=model, name=NAME, 
            loaders={'standard': test_loader_standard},
            metas={'standard': test_meta_standard})


if DATA_TYPE == 'segmented' or DATA_TYPE == 'all':
    train_loader_segmented = create_triplet_loader(
        train_windows_segmented, train_meta_segmented['classes'], train_meta_segmented['subjects'],
        batch=BATCH_SIZE, n_classes=CLASSES, n_subjects=N_SUBJECTS)
    val_loader_triplet_segmented = create_triplet_loader(
        val_windows_segmented, val_meta_segmented['classes'], val_meta_segmented['subjects'],
        batch=BATCH_SIZE, n_classes=CLASSES, n_subjects=26)

    NAME = f"cnn_segmented_trp"
    model = CNN()
    print(model, f"\nParameters count: {count_params(model):,}")

    if MODE == "train":
        train_triplet(model=model, name=NAME, 
            train_loader=train_loader_segmented, val_loader=val_loader_triplet_segmented, 
            criterion_ce=nn.CrossEntropyLoss(weight=None),
            criterion_tri=TripletLoss(margin= MARGIN, batch_hard=True), 
            save_chkp=SAVE_CHKP, epochs=EPOCHS, lr=LR_INIT, min_lr=LR_MIN, 
            lr_factor=LR_FACTOR, lr_patience=LR_PATIENCE, patience=PATIENCE,
            alpha_start=ALPHA_START, alpha_end=ALPHA_END, warmup_epochs=WARMUP)
        torch.save(model.state_dict(), join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
    else:
        model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))

    eval_test(model=model, name=NAME, 
            loaders={'segmented': test_loader_segmented},
            metas={'segmented': test_meta_segmented, })
