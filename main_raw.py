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


# ======== PIPELINE ========
train_loader = create_loader(train_windows, train_meta['classes'], 
                            batch=BATCH_SIZE, shuffle=True)
val_loader = create_loader(val_windows, val_meta['classes'], 
                            batch=BATCH_SIZE, shuffle=False)
test_loader = create_loader(test_windows, test_meta['classes'], 
                            batch=BATCH_SIZE, shuffle=False)

train_loader_segmented = create_loader(train_windows_segmented, 
                            train_meta_segmented['classes'], 
                            batch=BATCH_SIZE, shuffle=True)
val_loader_segmented = create_loader(val_windows_segmented, 
                            val_meta_segmented['classes'], 
                            batch=BATCH_SIZE, shuffle=False)
test_loader_segmented = create_loader(test_windows_segmented, 
                            test_meta_segmented['classes'], 
                            batch=BATCH_SIZE, shuffle=False)

train_loader_relabeled = create_loader(train_windows_relabeled, 
                            train_meta_relabeled['classes'],
                            batch=BATCH_SIZE, shuffle=True)
val_loader_relabeled = create_loader(val_windows_relabeled, 
                            val_meta_relabeled['classes'],
                            batch=BATCH_SIZE, shuffle=False)
test_loader_relabeled = create_loader(test_windows_relabeled, 
                            test_meta_relabeled['classes'],
                            batch=BATCH_SIZE, shuffle=False)


# NAME = "cnn_raw"
# model = CNN()
# print(model, f"\nParameters count: {count_params(model):,}")
# weights = torch.tensor(compute_class_weight('balanced', 
#                                classes=np.arange(CLASSES), 
#                                 y=train_meta['classes']),
#                                 dtype=torch.float32,
#                                 device=DEVICE)
# train(model=model, name=NAME, 
#       train_loader=train_loader,
#       val_loader=val_loader, 
#       loss_fn=nn.CrossEntropyLoss(weight=weights),
#       save_chkp=SAVE_CHKP)
# torch.save(model.state_dict(), join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
# # model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))
# eval_test(model=model, name=NAME, 
#           loaders={'raw': test_loader, 
#                    'segmented': test_loader_segmented, 
#                    'relabeled': test_loader_relabeled},
#            metas={'raw': test_meta, 
#                   'segmented': test_meta_segmented, 
#                   'relabeled': test_meta_relabeled})


NAME = "cnn_raw_eq"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
weights = torch.tensor(compute_class_weight('balanced', 
                               classes=np.arange(CLASSES), 
                                y=train_meta['classes']),
                                dtype=torch.float32,
                                device=DEVICE)
train(model=model, name=NAME, 
      train_loader=train_loader,
      val_loader=val_loader, 
      loss_fn=EqLoss(weight=weights),
      save_chkp=SAVE_CHKP)
torch.save(model.state_dict(), join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
# model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, "f{NAME}.pt")))
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader, 
                   'segmented': test_loader_segmented, 
                   'relabeled': test_loader_relabeled},
           metas={'raw': test_meta, 
                  'segmented': test_meta_segmented, 
                  'relabeled': test_meta_relabeled})


# NAME = "cnn_raw_rest"
# model = CNN()
# print(model, f"\nParameters count: {count_params(model):,}")
# weights = torch.tensor(compute_class_weight('balanced', 
#                                classes=np.arange(CLASSES), 
#                                 y=train_meta['classes']),
#                                 dtype=torch.float32,
#                                 device=DEVICE)
# train(model=model, name=NAME, 
#       train_loader=train_loader,
#       val_loader=val_loader, 
#       loss_fn=RestLoss(1.0, 1.0, weight=weights),
#       save_chkp=SAVE_CHKP)
# torch.save(model.state_dict(), join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
# # model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, "f{NAME}.pt")))
# eval_test(model=model, name=NAME, 
#           loaders={'raw': test_loader, 
#                    'segmented': test_loader_segmented, 
#                    'relabeled': test_loader_relabeled},
#            metas={'raw': test_meta, 
#                   'segmented': test_meta_segmented, 
#                   'relabeled': test_meta_relabeled})
