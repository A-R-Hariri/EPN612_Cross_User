import warnings, sys, os, gc
from os.path import join
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"

import torch; print(torch.cuda.is_available())
from torchview import draw_graph
import cairosvg

import libemg
from libemg.datasets import get_dataset_list
from libemg.feature_extractor import FeatureExtractor

import numpy as np, pandas as pd
from scipy import stats

import libemg
from libemg.datasets import get_dataset_list
from libemg.feature_extractor import FeatureExtractor

import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA
import seaborn as sns
from numpy.lib.stride_tricks import sliding_window_view
import random, copy, time, json
import plotly.express as px
import re, glob
from sklearn.utils.class_weight import compute_class_weight
from statsmodels.stats.multitest import multipletests
from scipy import stats
from itertools import combinations
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import *
from models import *


SEED = 13; random.seed(SEED); np.random.seed(SEED)
GENERATOR = torch.manual_seed(SEED)
MMAP_MODE = 'r'; SAVE_CHKP = True


TAG = sys.argv[2] if len(sys.argv) > 2 else "raw"
LOSS = sys.argv[3] if len(sys.argv) > 3 else "base"
NAME = f'cross_mhcnn_{TAG}_{LOSS}'


train_windows = np.load(join(PICKLE_PATH, f'train_windows_{TAG}.npy'), mmap_mode=MMAP_MODE)
train_meta = np.load(join(PICKLE_PATH, f'train_meta_{TAG}.npy'), allow_pickle=True).item()
val_windows = np.load(join(PICKLE_PATH, f'val_windows_{TAG}.npy'), mmap_mode=MMAP_MODE)
val_meta = np.load(join(PICKLE_PATH, f'val_meta_{TAG}.npy'), allow_pickle=True).item()
test_windows_raw = np.load(join(PICKLE_PATH, f'test_windows_raw.npy'), mmap_mode=MMAP_MODE)
test_meta_raw = np.load(join(PICKLE_PATH, f'test_meta_raw.npy'), allow_pickle=True).item()

test_loader_raw = create_loader(test_windows_raw, 
                                test_meta_raw['classes'], 
                                test_meta_raw['subjects'], 
                                batch=BATCH_SIZE, shuffle=False)


model = MHCNN() if 'grl' not in LOSS else MHCNN_GRL()
run_pca_sweep(model, test_loader_raw, NAME, dims=4)