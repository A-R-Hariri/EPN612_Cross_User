import warnings, sys, os, gc
from os.path import join
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"

import torch; print(torch.cuda.is_available())

import numpy as np
import random
from sklearn.utils.class_weight import compute_class_weight

from utils import *
from models import *


SEED = 13; random.seed(SEED); np.random.seed(SEED)
GENERATOR = torch.manual_seed(SEED)
MMAP_MODE = 'r'; SAVE_CHKP = True

TAG = sys.argv[2] if len(sys.argv) > 2 else "raw"
MODE = sys.argv[3] if len(sys.argv) > 3 else "train"


# ======== LOAD DATA ========
train_windows = np.load(join(PICKLE_PATH, f'train_windows_{TAG}.npy'), mmap_mode=MMAP_MODE)
train_meta = np.load(join(PICKLE_PATH, f'train_meta_{TAG}.npy'), allow_pickle=True).item()
val_windows = np.load(join(PICKLE_PATH, f'val_windows_{TAG}.npy'), mmap_mode=MMAP_MODE)
val_meta = np.load(join(PICKLE_PATH, f'val_meta_{TAG}.npy'), allow_pickle=True).item()

test_windows_raw = np.load(join(PICKLE_PATH, f'test_windows_raw.npy'), mmap_mode=MMAP_MODE)
test_meta_raw = np.load(join(PICKLE_PATH, f'test_meta_raw.npy'), allow_pickle=True).item()
test_windows_standard = np.load(join(PICKLE_PATH, 'test_windows_standard.npy'), mmap_mode=MMAP_MODE)
test_meta_standard = np.load(join(PICKLE_PATH, 'test_meta_standard.npy'), allow_pickle=True).item()
test_windows_segmented = np.load(join(PICKLE_PATH, 'test_windows_segmented.npy'), mmap_mode=MMAP_MODE)
test_meta_segmented = np.load(join(PICKLE_PATH, 'test_meta_segmented.npy'), allow_pickle=True).item()
test_windows_relabeled = np.load(join(PICKLE_PATH, 'test_windows_relabeled.npy'), mmap_mode=MMAP_MODE)
test_meta_relabeled = np.load(join(PICKLE_PATH, 'test_meta_relabeled.npy'), allow_pickle=True).item()


# ======== PIPELINE ========
train_loader = create_loader(train_windows, 
                             train_meta['classes'], 
                             train_meta['subjects'],
                             batch=BATCH_SIZE, shuffle=True)
val_loader = create_loader(val_windows,
                           val_meta['classes'], 
                           val_meta['subjects'],
                           batch=BATCH_SIZE, shuffle=True)

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

weights = torch.tensor(compute_class_weight('balanced', 
                                classes=np.arange(CLASSES), 
                                y=train_meta['classes']),
                                dtype=torch.float32,
                                device=DEVICE)


NAME = f"cnn_{TAG}_base"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train(model=model, name=NAME, 
          train_loader=train_loader,
          val_loader=val_loader, 
          loss_fn=nn.CrossEntropyLoss(weight=weights),
          save_chkp=SAVE_CHKP)
else:
    model.load_state_dict(torch.load(join
        (CHECKPOINT_PATH, NAME, f"{NAME}.pt"))['model_state_dict'])
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader_raw,
                   'standard': test_loader_standard,
                   'segmented': test_loader_segmented,
                   'relabeled': test_loader_relabeled},
           metas={'raw': test_meta_raw,
                  'standard': test_meta_standard,
                  'segmented': test_meta_segmented,
                  'relabeled': test_meta_relabeled})


NAME = f"cnn_{TAG}_std"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train(model=model, name=NAME, 
          train_loader=train_loader,
          val_loader=val_loader, 
          loss_fn=STDLoss(),
          save_chkp=SAVE_CHKP)
else:
    model.load_state_dict(torch.load(join
        (CHECKPOINT_PATH, NAME, f"{NAME}.pt"))['model_state_dict'])
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader_raw,
                   'standard': test_loader_standard,
                   'segmented': test_loader_segmented,
                   'relabeled': test_loader_relabeled},
           metas={'raw': test_meta_raw,
                  'standard': test_meta_standard,
                  'segmented': test_meta_segmented,
                  'relabeled': test_meta_relabeled})


NAME = f"cnn_{TAG}_cvar"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train(model=model, name=NAME, 
          train_loader=train_loader,
          val_loader=val_loader, 
          loss_fn=CVaRLoss(weight=weights),
          save_chkp=SAVE_CHKP)
else:
    model.load_state_dict(torch.load(join
        (CHECKPOINT_PATH, NAME, f"{NAME}.pt"))['model_state_dict'])
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader_raw,
                   'standard': test_loader_standard,
                   'segmented': test_loader_segmented,
                   'relabeled': test_loader_relabeled},
           metas={'raw': test_meta_raw,
                  'standard': test_meta_standard,
                  'segmented': test_meta_segmented,
                  'relabeled': test_meta_relabeled})


NAME = f"cnn_{TAG}_rest"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train(model=model, name=NAME, 
          train_loader=train_loader,
          val_loader=val_loader, 
          loss_fn=RestLoss(weight=weights),
          save_chkp=SAVE_CHKP)
else:
    model.load_state_dict(torch.load(join
        (CHECKPOINT_PATH, NAME, f"{NAME}.pt"))['model_state_dict'])
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader_raw,
                   'standard': test_loader_standard,
                   'segmented': test_loader_segmented,
                   'relabeled': test_loader_relabeled},
           metas={'raw': test_meta_raw,
                  'standard': test_meta_standard,
                  'segmented': test_meta_segmented,
                  'relabeled': test_meta_relabeled})


NAME = f"cnn_{TAG}_act"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train(model=model, name=NAME, 
          train_loader=train_loader,
          val_loader=val_loader, 
          loss_fn=ActiveLoss(weight=weights),
          save_chkp=SAVE_CHKP)
else:
    model.load_state_dict(torch.load(join
        (CHECKPOINT_PATH, NAME, f"{NAME}.pt"))['model_state_dict'])
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader_raw,
                   'standard': test_loader_standard,
                   'segmented': test_loader_segmented,
                   'relabeled': test_loader_relabeled},
           metas={'raw': test_meta_raw,
                  'standard': test_meta_standard,
                  'segmented': test_meta_segmented,
                  'relabeled': test_meta_relabeled})


NAME = f"cnn_{TAG}_grl"
model = CNN_GRL()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train_grl(model=model, name=NAME, 
          train_loader=train_loader,
          val_loader=val_loader, 
          loss_fn=nn.CrossEntropyLoss(weight=weights),
          loss_fn_sbj=nn.CrossEntropyLoss(),
          save_chkp=SAVE_CHKP)
else:
    model.load_state_dict(torch.load(join
        (CHECKPOINT_PATH, NAME, f"{NAME}.pt"))['model_state_dict'])
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader_raw,
                   'standard': test_loader_standard,
                   'segmented': test_loader_segmented,
                   'relabeled': test_loader_relabeled},
           metas={'raw': test_meta_raw,
                  'standard': test_meta_standard,
                  'segmented': test_meta_segmented,
                  'relabeled': test_meta_relabeled})


NAME = f"cnn_{TAG}_sbj"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train_sbj(model=model, name=NAME, 
          train_loader=train_loader,
          val_loader=val_loader, 
          loss_fn=PerSubjectLoss(weight=weights),
          save_chkp=SAVE_CHKP)
else:
    model.load_state_dict(torch.load(join
        (CHECKPOINT_PATH, NAME, f"{NAME}.pt"))['model_state_dict'])
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader_raw,
                   'standard': test_loader_standard,
                   'segmented': test_loader_segmented,
                   'relabeled': test_loader_relabeled},
           metas={'raw': test_meta_raw,
                  'standard': test_meta_standard,
                  'segmented': test_meta_segmented,
                  'relabeled': test_meta_relabeled})


NAME = f"cnn_{TAG}_proto"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train(model=model, name=NAME, 
          train_loader=train_loader,
          val_loader=val_loader, 
          loss_fn=PrototypeLoss(weight=weights),
          return_emb=True, return_logits=True,
          save_chkp=SAVE_CHKP)
else:
    model.load_state_dict(torch.load(join
        (CHECKPOINT_PATH, NAME, f"{NAME}.pt"))['model_state_dict'])
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader_raw,
                   'standard': test_loader_standard,
                   'segmented': test_loader_segmented,
                   'relabeled': test_loader_relabeled},
           metas={'raw': test_meta_raw,
                  'standard': test_meta_standard,
                  'segmented': test_meta_segmented,
                  'relabeled': test_meta_relabeled})


NAME = f"cnn_{TAG}_1va"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train(model=model, name=NAME, 
          train_loader=train_loader,
          val_loader=val_loader, 
          loss_fn=OneVsAllLoss(weight=weights),
          return_emb=True, return_logits=True,
          save_chkp=SAVE_CHKP)
else:
    model.load_state_dict(torch.load(join
        (CHECKPOINT_PATH, NAME, f"{NAME}.pt"))['model_state_dict'])
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader_raw,
                   'standard': test_loader_standard,
                   'segmented': test_loader_segmented,
                   'relabeled': test_loader_relabeled},
           metas={'raw': test_meta_raw,
                  'standard': test_meta_standard,
                  'segmented': test_meta_segmented,
                  'relabeled': test_meta_relabeled})


NAME = f"cnn_{TAG}_ang"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train(model=model, name=NAME, 
          train_loader=train_loader,
          val_loader=val_loader, 
          loss_fn=AngularLoss(weight=weights),
          return_emb=True, return_logits=True,
          save_chkp=SAVE_CHKP)
else:
    model.load_state_dict(torch.load(join
        (CHECKPOINT_PATH, NAME, f"{NAME}.pt"))['model_state_dict'])
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader_raw,
                   'standard': test_loader_standard,
                   'segmented': test_loader_segmented,
                   'relabeled': test_loader_relabeled},
           metas={'raw': test_meta_raw,
                  'standard': test_meta_standard,
                  'segmented': test_meta_segmented,
                  'relabeled': test_meta_relabeled})


# c = np.unique(train_meta['classes'], return_counts=True)[1]
# indx_t = np.random.random(len(train_meta['classes'])) < c[np.argsort(c)[:-1]].mean() / c.max()
# train_loader = create_loader(train_windows[indx_t], 
#                              train_meta['classes'][indx_t], 
#                              train_meta['subjects'][indx_t],
#                              batch=BATCH_SIZE, shuffle=True)

# c = np.unique(val_meta['classes'], return_counts=True)[1]
# indx_v = np.random.random(len(val_meta['classes'])) < c[np.argsort(c)[:-1]].mean() / c.max()
# val_loader = create_loader(val_windows[indx_v],
#                            val_meta['classes'][indx_v], 
#                            val_meta['subjects'][indx_v],
#                            batch=BATCH_SIZE, shuffle=True)


# NAME = f"cnn_{TAG}_base-bal"
# model = CNN()
# print(model, f"\nParameters count: {count_params(model):,}")
# if MODE == "train":
#     train(model=model, name=NAME, 
#         train_loader=train_loader,
#         val_loader=val_loader, 
#         loss_fn=nn.CrossEntropyLoss(weight=None),
#         save_chkp=SAVE_CHKP)
# else:
#     model.load_state_dict(torch.load(join
#         (CHECKPOINT_PATH, NAME, f"{NAME}.pt"))['model_state_dict'])
# eval_test(model=model, name=NAME, 
#           loaders={'raw': test_loader_raw,
#                    'standard': test_loader_standard,
#                    'segmented': test_loader_segmented,
#                    'relabeled': test_loader_relabeled},
#            metas={'raw': test_meta_raw,
#                   'standard': test_meta_standard,
#                   'segmented': test_meta_segmented,
#                   'relabeled': test_meta_relabeled})

    
# NAME = f"cnn_{TAG}_std-bal"
# model = CNN()
# print(model, f"\nParameters count: {count_params(model):,}")
# if MODE == "train":
#     train(model=model, name=NAME, 
#         train_loader=train_loader,
#         val_loader=val_loader, 
#         loss_fn=STDLoss(),
#         save_chkp=SAVE_CHKP)
# else:
#     model.load_state_dict(torch.load(join
#         (CHECKPOINT_PATH, NAME, f"{NAME}.pt"))['model_state_dict'])
# eval_test(model=model, name=NAME, 
#           loaders={'raw': test_loader_raw,
#                    'standard': test_loader_standard,
#                    'segmented': test_loader_segmented,
#                    'relabeled': test_loader_relabeled},
#            metas={'raw': test_meta_raw,
#                   'standard': test_meta_standard,
#                   'segmented': test_meta_segmented,
#                   'relabeled': test_meta_relabeled})


# NAME = f"cnn_{TAG}_rest-bal"
# model = CNN()
# print(model, f"\nParameters count: {count_params(model):,}")
# if MODE == "train":
#     train(model=model, name=NAME, 
#         train_loader=train_loader,
#         val_loader=val_loader, 
#         loss_fn=RestLoss(weight=None),
#         save_chkp=SAVE_CHKP)
# else:
#     model.load_state_dict(torch.load(join
#         (CHECKPOINT_PATH, NAME, f"{NAME}.pt"))['model_state_dict'])
# eval_test(model=model, name=NAME, 
#           loaders={'raw': test_loader_raw,
#                    'standard': test_loader_standard,
#                    'segmented': test_loader_segmented,
#                    'relabeled': test_loader_relabeled},
#            metas={'raw': test_meta_raw,
#                   'standard': test_meta_standard,
#                   'segmented': test_meta_segmented,
#                   'relabeled': test_meta_relabeled})