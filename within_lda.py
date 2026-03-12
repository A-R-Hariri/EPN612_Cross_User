import warnings, sys, os, gc
from os.path import join
warnings.filterwarnings("ignore")

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA)

import libemg
from libemg.datasets import get_dataset_list
from libemg.feature_extractor import FeatureExtractor

import numpy as np, pandas as pd
import random, copy, time
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from utils import *


WORKERS = 0; PRE_FETCH = 2; VERBOSE=False; BATCH_SIZE=128
PRESIST_WORKER = False; PIN_MEMORY = True

SEED = 13; random.seed(SEED); np.random.seed(SEED)
GENERATOR = torch.manual_seed(SEED)
MMAP_MODE = 'r'; SAVE_CHKP = False

SAMPLING_RATE = 200
FEATURE_LIST = ['WENG']
FEATURE_DIC = {'WENG_fs': SAMPLING_RATE}


# ======== LOAD DATA ========
train_data = np.load(join(PICKLE_PATH, 'train_data.npy'), allow_pickle=True).item()
val_data = np.load(join(PICKLE_PATH, 'val_data.npy'), allow_pickle=True).item()
test_data = np.load(join(PICKLE_PATH, 'test_data.npy'), allow_pickle=True).item()

train_windows = np.load(join(PICKLE_PATH, 'train_windows.npy'), mmap_mode=MMAP_MODE)
train_meta = np.load(join(PICKLE_PATH, 'train_meta.npy'), allow_pickle=True).item()
val_windows = np.load(join(PICKLE_PATH, 'val_windows.npy'), mmap_mode=MMAP_MODE)
val_meta = np.load(join(PICKLE_PATH, 'val_meta.npy'), allow_pickle=True).item()
test_windows = np.load(join(PICKLE_PATH, 'test_windows.npy'), mmap_mode=MMAP_MODE)
test_meta = np.load(join(PICKLE_PATH, 'test_meta.npy'), allow_pickle=True).item()

# Within
REPS = sys.argv[1].split(',') if len(sys.argv) > 1 else 15
REPS = list(map(int, REPS))

for rep in REPS:
    NAME = f'lda_raw_within_{rep}'
    results = []

    ranges = [(0, 306), (306, 332), (332, 612)]
    data_list = [train_data, val_data, test_data]

    for d, r in enumerate(ranges):
        for i in range(*r):
            print(i)

            data_s = data_list[d].isolate_data("subjects", [i], fast=True)

            data = data_s.isolate_data("reps", list(range(rep)), fast=True)
            train_windows, train_meta = data.parse_windows(SEQ, INC)

            data = data_s.isolate_data("reps", list(range(15, 20)), fast=True)
            val_windows, val_meta = data.parse_windows(SEQ, INC)

            data = data_s.isolate_data("reps", list(range(20, 25)), fast=True)
            test_windows, test_meta = data.parse_windows(SEQ, INC)

            weights = torch.tensor(compute_class_weight('balanced', 
                                        classes=np.arange(CLASSES), 
                                            y=train_meta['classes']),
                                            dtype=torch.float32,
                                            device=DEVICE)

            # -------- Features --------
            feature_extractor = FeatureExtractor()
            
            train_windows = feature_extractor.extract_features(FEATURE_LIST, train_windows, array=True,
                                        fix_feature_errors=False, feature_dic=FEATURE_DIC).reshape((
                                            train_windows.shape[0], -1))
            val_windows = feature_extractor.extract_features(FEATURE_LIST, val_windows, array=True,
                                        fix_feature_errors=False, feature_dic=FEATURE_DIC).reshape((
                                            val_windows.shape[0], -1))
            test_windows = feature_extractor.extract_features(FEATURE_LIST, test_windows, array=True,
                                        fix_feature_errors=False, feature_dic=FEATURE_DIC).reshape((
                                            test_windows.shape[0], -1))
            
            model = LDA()
            model = model.fit(train_windows, train_meta['classes'])

            _result = eval_within_lda(model=model,
                                    x=test_windows.reshape((test_windows.shape[0], -1)),
                                    meta=test_meta)
            results.append(_result)
            print(_result['acc_mean'])

            del train_windows, val_windows, test_windows, model
            gc.collect()

    os.makedirs(f"{CHECKPOINT_PATH}", exist_ok=True)
    os.makedirs(f"{CHECKPOINT_PATH}/{NAME}/", exist_ok=True)
    np.save(f"{CHECKPOINT_PATH}/{NAME}/results.npy", results)