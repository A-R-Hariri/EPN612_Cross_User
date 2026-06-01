import warnings, sys, os, gc
from os.path import join
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"

import torch; print(torch.cuda.is_available())

import numpy as np
import random

from utils import *
from models import *


SEED = 13; random.seed(SEED); np.random.seed(SEED)
GENERATOR = torch.manual_seed(SEED)
MMAP_MODE = 'r'; SAVE_CHKP = True

RUN = ''
N_SUBJECTS = 306; MARGIN = 0.5; W_HARD = 1.0; W_SOFT = 0.0
ALPHA_START = 0.01; ALPHA_END = 0.25; WARMUP = 25


# ======== LOAD DATA ========
train_windows_raw = np.load(join(PICKLE_PATH, 'train_windows_raw.npy'), mmap_mode=MMAP_MODE)
train_meta_raw = np.load(join(PICKLE_PATH, 'train_meta_raw.npy'), allow_pickle=True).item()
val_windows_raw = np.load(join(PICKLE_PATH, 'val_windows_raw.npy'), mmap_mode=MMAP_MODE)
val_meta_raw = np.load(join(PICKLE_PATH, 'val_meta_raw.npy'), allow_pickle=True).item()
test_windows_raw = np.load(join(PICKLE_PATH, 'test_windows_raw.npy'), mmap_mode=MMAP_MODE)
test_meta_raw = np.load(join(PICKLE_PATH, 'test_meta_raw.npy'), allow_pickle=True).item()

train_windows_standard = np.load(join(PICKLE_PATH, 'train_windows_standard.npy'), mmap_mode=MMAP_MODE)
train_meta_standard = np.load(join(PICKLE_PATH, 'train_meta_standard.npy'), allow_pickle=True).item()
val_windows_standard = np.load(join(PICKLE_PATH, 'val_windows_standard.npy'), mmap_mode=MMAP_MODE)
val_meta_standard = np.load(join(PICKLE_PATH, 'val_meta_standard.npy'), allow_pickle=True).item()
test_windows_standard = np.load(join(PICKLE_PATH, 'test_windows_standard.npy'), mmap_mode=MMAP_MODE)
test_meta_standard = np.load(join(PICKLE_PATH, 'test_meta_standard.npy'), allow_pickle=True).item()

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


# ======== PIPELINE ========
test_loader_raw = create_loader(test_windows_raw, 
                                test_meta_raw['classes'], 
                                test_meta_raw['subjects'], 
                                batch=BATCH_SIZE, shuffle=False)
test_loader_standard = create_loader(test_windows_standard, 
                                     test_meta_standard['classes'],
                                     test_meta_standard['subjects'],
                                     batch=BATCH_SIZE, shuffle=False)
test_loader_segmented = create_loader(test_windows_segmented, 
                                      test_meta_segmented['classes'], 
                                      test_meta_segmented['subjects'], 
                                      batch=BATCH_SIZE, shuffle=False)
test_loader_relabeled = create_loader(test_windows_relabeled, 
                                      test_meta_relabeled['classes'],
                                      test_meta_relabeled['subjects'],
                                      batch=BATCH_SIZE, shuffle=False)


TAG = sys.argv[2] if len(sys.argv) > 2 else "all"
MODE = sys.argv[3] if len(sys.argv) > 3 else "train"


if TAG == 'raw' or TAG == 'all':
    train_loader = create_triplet_loader(
        train_windows_raw, train_meta_raw['classes'], train_meta_raw['subjects'],
        batch=BATCH_SIZE, n_classes=CLASSES, n_subjects=N_SUBJECTS)
    val_loader_triplet = create_triplet_loader(
        val_windows_raw, val_meta_raw['classes'], val_meta_raw['subjects'],
        batch=BATCH_SIZE, n_classes=CLASSES, n_subjects=26)

    NAME = f"mhcnn_raw_trp{RUN}"
    model = MHCNN()
    print(model, f"\nParameters count: {count_params(model):,}")

    if MODE == "train":
        train_triplet(model=model, name=NAME, 
            train_loader=train_loader, val_loader=val_loader_triplet, 
            criterion_ce=nn.CrossEntropyLoss(weight=None),
            criterion_tri=TripletLoss(margin=MARGIN, batch_hard=True,
                                        w_hard=W_HARD, w_soft=W_SOFT), 
            save_chkp=SAVE_CHKP, epochs=EPOCHS, lr=LR_INIT, min_lr=LR_MIN, 
            lr_factor=LR_FACTOR, lr_patience=LR_PATIENCE, patience=PATIENCE,
            alpha_start=ALPHA_START, alpha_end=ALPHA_END, warmup_epochs=WARMUP)
    else:
        model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))

    eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader_raw,
                   'standard': test_loader_standard,
                   'segmented': test_loader_segmented,
                   'relabeled': test_loader_relabeled},
           metas={'raw': test_meta_raw,
                  'standard': test_meta_standard,
                  'segmented': test_meta_segmented,
                  'relabeled': test_meta_relabeled})


# if TAG == 'standard' or TAG == 'all':
#     train_loader_standard = create_triplet_loader(
#         train_windows_standard, train_meta_standard['classes'], train_meta_standard['subjects'],
#         batch=BATCH_SIZE, n_classes=CLASSES, n_subjects=N_SUBJECTS)
#     val_loader_triplet_standard = create_triplet_loader(
#         val_windows_standard, val_meta_standard['classes'], val_meta_standard['subjects'],
#         batch=BATCH_SIZE, n_classes=CLASSES, n_subjects=26)

#     NAME = f"mhcnn_standard_trp{RUN}"
#     model = MHCNN()
#     print(model, f"\nParameters count: {count_params(model):,}")

#     if MODE == "train":
#         train_triplet(model=model, name=NAME, 
#             train_loader=train_loader_standard, val_loader=val_loader_triplet_standard, 
#             criterion_ce=nn.CrossEntropyLoss(weight=None),
#             criterion_tri=TripletLoss(margin=MARGIN, batch_hard=True,
#                                         w_hard=W_HARD, w_soft=W_SOFT), 
#             save_chkp=SAVE_CHKP, epochs=EPOCHS, lr=LR_INIT, min_lr=LR_MIN, 
#             lr_factor=LR_FACTOR, lr_patience=LR_PATIENCE, patience=PATIENCE,
#             alpha_start=ALPHA_START, alpha_end=ALPHA_END, warmup_epochs=WARMUP)
#     else:
#         model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))

#     eval_test(model=model, name=NAME, 
#           loaders={'raw': test_loader_raw,
#                    'standard': test_loader_standard,
#                    'segmented': test_loader_segmented,
#                    'relabeled': test_loader_relabeled},
#            metas={'raw': test_meta_raw,
#                   'standard': test_meta_standard,
#                   'segmented': test_meta_segmented,
#                   'relabeled': test_meta_relabeled})


if TAG == 'segmented' or TAG == 'all':
    train_loader_segmented = create_triplet_loader(
        train_windows_segmented, train_meta_segmented['classes'], train_meta_segmented['subjects'],
        batch=BATCH_SIZE, n_classes=CLASSES, n_subjects=N_SUBJECTS)
    val_loader_triplet_segmented = create_triplet_loader(
        val_windows_segmented, val_meta_segmented['classes'], val_meta_segmented['subjects'],
        batch=BATCH_SIZE, n_classes=CLASSES, n_subjects=26)

    NAME = f"mhcnn_segmented_trp{RUN}"
    model = MHCNN()
    print(model, f"\nParameters count: {count_params(model):,}")

    if MODE == "train":
        train_triplet(model=model, name=NAME, 
            train_loader=train_loader_segmented, val_loader=val_loader_triplet_segmented, 
            criterion_ce=nn.CrossEntropyLoss(weight=None),
            criterion_tri=TripletLoss(margin=MARGIN, batch_hard=True,
                                        w_hard=W_HARD, w_soft=W_SOFT), 
            save_chkp=SAVE_CHKP, epochs=EPOCHS, lr=LR_INIT, min_lr=LR_MIN, 
            lr_factor=LR_FACTOR, lr_patience=LR_PATIENCE, patience=PATIENCE,
            alpha_start=ALPHA_START, alpha_end=ALPHA_END, warmup_epochs=WARMUP)
    else:
        model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))

    eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader_raw,
                   'standard': test_loader_standard,
                   'segmented': test_loader_segmented,
                   'relabeled': test_loader_relabeled},
           metas={'raw': test_meta_raw,
                  'standard': test_meta_standard,
                  'segmented': test_meta_segmented,
                  'relabeled': test_meta_relabeled})


if TAG == 'relabeled' or TAG == 'all':
    train_loader_relabeled = create_triplet_loader(
        train_windows_relabeled, train_meta_relabeled['classes'], train_meta_relabeled['subjects'],
        batch=BATCH_SIZE, n_classes=CLASSES, n_subjects=N_SUBJECTS)
    val_loader_triplet_relabeled = create_triplet_loader(
        val_windows_relabeled, val_meta_relabeled['classes'], val_meta_relabeled['subjects'],
        batch=BATCH_SIZE, n_classes=CLASSES, n_subjects=26)

    NAME = f"mhcnn_relabeled_trp{RUN}"
    model = MHCNN()
    print(model, f"\nParameters count: {count_params(model):,}")

    if MODE == "train":
        train_triplet(model=model, name=NAME, 
            train_loader=train_loader_relabeled, val_loader=val_loader_triplet_relabeled, 
            criterion_ce=nn.CrossEntropyLoss(weight=None),
            criterion_tri=TripletLoss(margin=MARGIN, batch_hard=True,
                                        w_hard=W_HARD, w_soft=W_SOFT), 
            save_chkp=SAVE_CHKP, epochs=EPOCHS, lr=LR_INIT, min_lr=LR_MIN, 
            lr_factor=LR_FACTOR, lr_patience=LR_PATIENCE, patience=PATIENCE,
            alpha_start=ALPHA_START, alpha_end=ALPHA_END, warmup_epochs=WARMUP)
    else:
        model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))

    eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader_raw,
                   'standard': test_loader_standard,
                   'segmented': test_loader_segmented,
                   'relabeled': test_loader_relabeled},
           metas={'raw': test_meta_raw,
                  'standard': test_meta_standard,
                  'segmented': test_meta_segmented,
                  'relabeled': test_meta_relabeled})