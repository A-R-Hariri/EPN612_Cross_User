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

# train_windows_segmented = np.load(join(PICKLE_PATH, 'train_windows_segmented.npy'), mmap_mode=MMAP_MODE)
# train_meta_segmented = np.load(join(PICKLE_PATH, 'train_meta_segmented.npy'), allow_pickle=True).item()
# val_windows_segmented = np.load(join(PICKLE_PATH, 'val_windows_segmented.npy'), mmap_mode=MMAP_MODE)
# val_meta_segmented = np.load(join(PICKLE_PATH, 'val_meta_segmented.npy'), allow_pickle=True).item()
test_windows_segmented = np.load(join(PICKLE_PATH, 'test_windows_segmented.npy'), mmap_mode=MMAP_MODE)
test_meta_segmented = np.load(join(PICKLE_PATH, 'test_meta_segmented.npy'), allow_pickle=True).item()

# train_windows_relabeled = np.load(join(PICKLE_PATH, 'train_windows_relabeled.npy'), mmap_mode=MMAP_MODE)
# train_meta_relabeled = np.load(join(PICKLE_PATH, 'train_meta_relabeled.npy'), allow_pickle=True).item()
# val_windows_relabeled = np.load(join(PICKLE_PATH, 'val_windows_relabeled.npy'), mmap_mode=MMAP_MODE)
# val_meta_relabeled = np.load(join(PICKLE_PATH, 'val_meta_relabeled.npy'), allow_pickle=True).item()
test_windows_relabeled = np.load(join(PICKLE_PATH, 'test_windows_relabeled.npy'), mmap_mode=MMAP_MODE)
test_meta_relabeled = np.load(join(PICKLE_PATH, 'test_meta_relabeled.npy'), allow_pickle=True).item()

# train_windows_standard = np.load(join(PICKLE_PATH, 'train_windows_standard.npy'), mmap_mode=MMAP_MODE)
# train_meta_standard = np.load(join(PICKLE_PATH, 'train_meta_standard.npy'), allow_pickle=True).item()
# val_windows_standard = np.load(join(PICKLE_PATH, 'val_windows_standard.npy'), mmap_mode=MMAP_MODE)
# val_meta_standard = np.load(join(PICKLE_PATH, 'val_meta_standard.npy'), allow_pickle=True).item()
test_windows_standard = np.load(join(PICKLE_PATH, 'test_windows_standard.npy'), mmap_mode=MMAP_MODE)
test_meta_standard = np.load(join(PICKLE_PATH, 'test_meta_standard.npy'), allow_pickle=True).item()


# ======== PIPELINE ========
val_loader = create_loader(val_windows, val_meta['classes'], 
    batch=BATCH_SIZE, shuffle=False)
test_loader = create_loader(test_windows, test_meta['classes'], 
    batch=BATCH_SIZE, shuffle=False)

base_ids = train_meta['subjects']

for i in range(1, 2):
    fname = f"{CHECKPOINT_PATH}/cnn_raw_within_15/results.npy"
    res = np.load(fname, allow_pickle=True)
    df = pd.DataFrame(res.tolist())
    idx = slice(0, 306)
    acc_mean = df['acc_mean'][idx]
    act_mean = df['act_acc_mean'][idx]
    bal_mean = df['bal_acc_mean'][idx]

    ranked_ids = list(np.argsort(bal_mean))
    if i:
        ranked_ids = list(reversed(ranked_ids))

    unique_ids = np.arange(306)
    mapping = dict(zip(unique_ids, ranked_ids))
    train_meta['subjects'] = np.vectorize(mapping.get)(base_ids.copy())

    for s in [1, 2, 4, 8, 16, 32, 64, 128, 196, 306]:
        for r in [1, 2, 4, 8, 16, 24, 32, 40, 50]:
            print("S, R:", s, r)
            NAME = f"inc_cnn_raw_s{s}_r{r}"
            indx = (np.isin(train_meta['subjects'], np.arange(s)) & 
                    np.isin(train_meta['reps'], np.arange(r)))
            X = train_windows[indx]
            y = train_meta['classes'][indx]

            train_loader = create_loader(X, y, 
                batch=BATCH_SIZE, shuffle=True)

            weights = torch.tensor(compute_class_weight('balanced', 
                classes=np.arange(CLASSES), 
                y=y),
                dtype=torch.float32,
                device=DEVICE)
                
            model = CNN()
            print(model, f"\nParameters count: {count_params(model):,}")
            train(model=model, name=NAME, 
                train_loader=train_loader,
                val_loader=val_loader, 
                loss_fn=nn.CrossEntropyLoss(weight=weights),
                save_chkp=False)

            eval_test(model=model, name=NAME, save=False,
                loaders={'raw': test_loader},
                csv_path=f'grid_ranked_{i}.csv',
                metas={'raw': test_meta})

            del train_loader, model
            torch.cuda.empty_cache()
            gc.collect()