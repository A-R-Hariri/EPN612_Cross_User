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
MMAP_MODE = 'r'; SAVE_CHKP = False


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
NAME = 'cnn_raw_within_lim'
results = []

ranges = [(0, 306), (306, 332), (332, 612)]
data_list = [train_data, val_data, test_data]

for d, r in enumerate(ranges):
    for i in range(*r):
        print(i)

        data_s = data_list[d].isolate_data("subjects", [i], fast=True)

        data = data_s.isolate_data("reps", list(range(1)), fast=True)
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

        train_loader = create_loader(train_windows, train_meta['classes'], 
                                    batch=128, shuffle=True)
        val_loader = create_loader(val_windows, val_meta['classes'], 
                                    batch=128, shuffle=False)
        test_loader = create_loader(test_windows, test_meta['classes'], 
                                    batch=128, shuffle=False)

        model = CNN()
        weights = torch.tensor(compute_class_weight('balanced', 
                                    classes=np.arange(CLASSES), 
                                        y=train_meta['classes']),
                                        dtype=torch.float32,
                                        device=DEVICE)
        train(model=model, name=NAME, 
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.CrossEntropyLoss(weight=weights),
            save_chkp=SAVE_CHKP, verbose=False)
        _result = eval_within(model=model, name=NAME, 
                            loader=test_loader,
                            meta=test_meta)
        results.append(_result)
        print(_result['acc_mean'])

        del train_loader, val_loader, test_loader, model
        torch.cuda.empty_cache()
        gc.collect()

os.makedirs(f"{CHECKPOINT_PATH}", exist_ok=True)
os.makedirs(f"{CHECKPOINT_PATH}/{NAME}/", exist_ok=True)
np.save(f"{CHECKPOINT_PATH}/{NAME}/results.npy", results)