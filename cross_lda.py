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
from models import *


SEED = 13; random.seed(SEED); np.random.seed(SEED)
GENERATOR = torch.manual_seed(SEED)
MMAP_MODE = 'r'; SAVE_CHKP = True

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

train_data_segmented = np.load(join(PICKLE_PATH, 'train_data_segmented.npy'), allow_pickle=True).item()
val_data_segmented = np.load(join(PICKLE_PATH, 'val_data_segmented.npy'), allow_pickle=True).item()
test_data_segmented = np.load(join(PICKLE_PATH, 'test_data_segmented.npy'), allow_pickle=True).item()

train_segmented_bounds = np.load(join(PICKLE_PATH, 'train_segmented_bounds.npy'))
val_segmented_bounds = np.load(join(PICKLE_PATH, 'val_segmented_bounds.npy'))
test_segmented_bounds = np.load(join(PICKLE_PATH, 'test_segmented_bounds.npy'))

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


# ======== FEATURES ========
feature_extractor = FeatureExtractor()

# train_windows = feature_extractor.extract_features(FEATURE_LIST, train_windows, array=True,
#                               fix_feature_errors=False, feature_dic=FEATURE_DIC).reshape((
#                                   train_windows.shape[0], -1))
# val_windows = feature_extractor.extract_features(FEATURE_LIST, val_windows, array=True,
#                               fix_feature_errors=False, feature_dic=FEATURE_DIC).reshape((
#                                   val_windows.shape[0], -1))
test_windows = feature_extractor.extract_features(FEATURE_LIST, test_windows, array=True,
                              fix_feature_errors=False, feature_dic=FEATURE_DIC).reshape((
                                  test_windows.shape[0], -1))

# train_windows_segmented = feature_extractor.extract_features(FEATURE_LIST, train_windows_segmented, array=True,
#                               fix_feature_errors=False, feature_dic=FEATURE_DIC).reshape((
#                                   train_windows_segmented.shape[0], -1))
# val_windows_segmented = feature_extractor.extract_features(FEATURE_LIST, val_windows_segmented, array=True,
#                               fix_feature_errors=False, feature_dic=FEATURE_DIC).reshape((
#                                   val_windows_segmented.shape[0], -1))
test_windows_segmented = feature_extractor.extract_features(FEATURE_LIST, test_windows_segmented, 
                                array=True, fix_feature_errors=False, feature_dic=FEATURE_DIC).reshape((
                                  test_windows_segmented.shape[0], -1))

# train_windows_relabeled = feature_extractor.extract_features(FEATURE_LIST, train_windows_relabeled, array=True,
#                               fix_feature_errors=False, feature_dic=FEATURE_DIC).reshape((
#                                   train_windows_relabeled.shape[0], -1))
# val_windows_relabeled = feature_extractor.extract_features(FEATURE_LIST, val_windows_relabeled, array=True,
#                               fix_feature_errors=False, feature_dic=FEATURE_DIC).reshape((
#                                   val_windows_relabeled.shape[0], -1))
# test_windows_relabeled = feature_extractor.extract_features(FEATURE_LIST, test_windows_relabeled, 
#                                 array=True, fix_feature_errors=False, feature_dic=FEATURE_DIC).reshape((
#                                   test_windows_relabeled.shape[0], -1))

train_windows_standard = feature_extractor.extract_features(FEATURE_LIST, train_windows_standard, array=True,
                              fix_feature_errors=False, feature_dic=FEATURE_DIC).reshape((
                                  train_windows_standard.shape[0], -1))
val_windows_standard = feature_extractor.extract_features(FEATURE_LIST, val_windows_standard, array=True,
                              fix_feature_errors=False, feature_dic=FEATURE_DIC).reshape((
                                  val_windows_standard.shape[0], -1))
test_windows_standard = feature_extractor.extract_features(FEATURE_LIST, test_windows_standard, 
                                array=True, fix_feature_errors=False, feature_dic=FEATURE_DIC).reshape((
                                  test_windows_standard.shape[0], -1))

n_features = test_windows_standard.shape[1]


# NAME = "lda_raw"
# model = LDA()
# model = model.fit(train_windows, train_meta['classes'])
# eval_test_lda(model=model, name=NAME, 
#           X={'raw': test_windows, 
#                    'segmented': test_windows_segmented, 
#                    'relabeled': test_windows_relabeled},
#            metas={'raw': test_meta, 
#                   'segmented': test_meta_segmented, 
#                   'relabeled': test_meta_relabeled})


# NAME = "lda_segmented"
# model = LDA()
# model = model.fit(train_windows_segmented, train_meta_segmented['classes'])
# eval_test_lda(model=model, name=NAME, 
#           X={'raw': test_windows, 
#                    'segmented': test_windows_segmented, 
#                    'relabeled': test_windows_relabeled},
#            metas={'raw': test_meta, 
#                   'segmented': test_meta_segmented, 
#                   'relabeled': test_meta_relabeled})


# NAME = "lda_relabeled"
# model = LDA()
# model = model.fit(train_windows_relabeled, train_meta_relabeled['classes'])
# eval_test_lda(model=model, name=NAME, 
#           X={'raw': test_windows, 
#                    'segmented': test_windows_segmented, 
#                    'relabeled': test_windows_relabeled},
#            metas={'raw': test_meta, 
#                   'segmented': test_meta_segmented, 
#                   'relabeled': test_meta_relabeled})


NAME = "lda_standard"
model = LDA()
model = model.fit(train_windows_standard, train_meta_standard['classes'])
eval_test_lda(model=model, name=NAME, 
          X={'raw': test_windows, 
                   'segmented': test_windows_segmented, 
                   'standard': test_windows_standard},
           metas={'raw': test_meta, 
                  'segmented': test_meta_segmented, 
                  'standard': test_meta_standard})