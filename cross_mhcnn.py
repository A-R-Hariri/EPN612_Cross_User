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


TAG = sys.argv[2] if len(sys.argv) > 2 else "raw"
NORM = sys.argv[3] == 'norm' if len(sys.argv) > 3 else False
_requested = set(sys.argv[4:]) if len(sys.argv) > 4 else {'all'}
MODE = sys.argv[5] if len(sys.argv) > 5 else "train"

SEED = 13; random.seed(SEED); np.random.seed(SEED)
GENERATOR = torch.manual_seed(SEED)
MMAP_MODE = 'r'; SAVE_CHKP = True

N_SUBJECTS = 306; MARGIN = 0.5; W_HARD = 1.0; W_SOFT = 0.0
ALPHA_START = 0.01; ALPHA_END = 0.25; WARMUP = 25
TAU = float('inf')


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


# ======== ORACLE NORMALIZE ========
if NORM:
    pop_mean, pop_std = population_channel_stats(train_windows)
    train_windows = normalize_per_user(train_windows, train_meta['subjects']) * 128.0
    val_windows = normalize_per_user(val_windows, val_meta['subjects']) * 128.0


# ======== LOADERS ========
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
                       dtype=torch.float32, device=DEVICE)
weights = None if TAG == 'raw' else weights


# ======== PIPELINE ========
_eval_loaders = dict(raw=test_loader_raw, standard=test_loader_standard,
                     segmented=test_loader_segmented, relabeled=test_loader_relabeled)
_eval_metas = dict(raw=test_meta_raw, standard=test_meta_standard,
                   segmented=test_meta_segmented, relabeled=test_meta_relabeled)
_eval_windows = dict(raw=(test_windows_raw, test_meta_raw),
                     standard=(test_windows_raw, test_meta_standard),
                     segmented=(test_windows_raw, test_meta_segmented),
                     relabeled=(test_windows_raw, test_meta_relabeled))

def _make_triplet_loaders():
    tl = create_triplet_loader(train_windows, train_meta['classes'], train_meta['subjects'],
                               batch=BATCH_SIZE, n_classes=CLASSES, n_subjects=N_SUBJECTS)
    vl = create_triplet_loader(val_windows, val_meta['classes'], val_meta['subjects'],
                               batch=BATCH_SIZE, n_classes=CLASSES, n_subjects=26)
    return tl, vl


# -------- model configs --------
CONFIGS = {
    'base': dict(model_cls=MHCNN, train_fn=train,
                 train_kwargs=dict(loss_fn=BaseLoss(weight=weights))),
    'std': dict(model_cls=MHCNN, train_fn=train,
                train_kwargs=dict(loss_fn=STDLoss())),
    'cvar': dict(model_cls=MHCNN, train_fn=train,
                 train_kwargs=dict(loss_fn=CVaRLoss(weight=weights))),
    'rest': dict(model_cls=MHCNN, train_fn=train,
                 train_kwargs=dict(loss_fn=RestLoss(weight=weights))),
    'act': dict(model_cls=MHCNN, train_fn=train,
                train_kwargs=dict(loss_fn=ActiveLoss(weight=weights))),
    'grl': dict(model_cls=MHCNN_GRL, train_fn=train_grl,
                train_kwargs=dict(loss_fn=BaseLoss(weight=weights),
                                  loss_fn_sbj=nn.CrossEntropyLoss())),
    'sbj': dict(model_cls=MHCNN, train_fn=train_sbj,
                train_kwargs=dict(loss_fn=PerSubjectLoss(weight=weights))),
    'proto': dict(model_cls=MHCNN, train_fn=train,
                  train_kwargs=dict(loss_fn=PrototypeLoss(weight=weights),
                                    return_emb=True, return_logits=True)),
    '1va': dict(model_cls=MHCNN, train_fn=train,
                train_kwargs=dict(loss_fn=OneVsAllLoss(weight=weights),
                                  return_emb=True, return_logits=True)),
    'ang': dict(model_cls=MHCNN, train_fn=train,
                train_kwargs=dict(loss_fn=AngularLoss(weight=weights),
                                  return_emb=True, return_logits=True)),
    'trp': dict(model_cls=MHCNN, train_fn=train_triplet,
                loader_fn=_make_triplet_loaders,
                ckpt_key=None,
                train_kwargs=dict(
                    criterion_ce=nn.CrossEntropyLoss(weight=None),
                    criterion_tri=TripletLoss(margin=MARGIN, batch_hard=True,
                                              w_hard=W_HARD, w_soft=W_SOFT),
                    epochs=EPOCHS, lr=LR_INIT, min_lr=LR_MIN,
                    lr_factor=LR_FACTOR, lr_patience=LR_PATIENCE,
                    patience=PATIENCE, alpha_start=ALPHA_START,
                    alpha_end=ALPHA_END, warmup_epochs=WARMUP)),
}


to_run = list(CONFIGS) if 'all' in _requested else [k for k in CONFIGS if k in _requested]

for variant in to_run:
    cfg = CONFIGS[variant]
    NAME = f"cross_mhcnn_{TAG}_{variant}" + ('-rn' if NORM else '')

    model = cfg['model_cls']()
    print(model, f"\nParameters count: {count_params(model):,}")

    if 'loader_fn' in cfg:
        torch.cuda.empty_cache()
        gc.collect()
        run_train_loader, run_val_loader = cfg['loader_fn']()
    else:
        run_train_loader, run_val_loader = train_loader, val_loader

    if MODE == 'train':
        cfg['train_fn'](model=model, name=NAME,
                        train_loader=run_train_loader,
                        val_loader=run_val_loader,
                        save_chkp=SAVE_CHKP,
                        **cfg['train_kwargs'])
    else:
        ckpt = torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
        key = cfg.get('ckpt_key', 'model_state_dict')
        model.load_state_dict(ckpt if key is None else ckpt[key])

    if not NORM:
        eval_test(model=model, name=NAME, loaders=_eval_loaders, metas=_eval_metas)
    else:
        eval_test_running(model, RunningNorm(CH, TAU, pop_mean, pop_std),
                          _eval_windows, NAME, SEED)