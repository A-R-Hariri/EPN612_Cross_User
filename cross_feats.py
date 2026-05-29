# ══════════════════════════════════════════════
# cross_feats.py
# ══════════════════════════════════════════════
import warnings, sys, os, gc
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from libemg.feature_extractor import FeatureExtractor
import numpy as np
import random
from os.path import join
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.class_weight import compute_class_weight

from utils import *
from models import *

# config
MODE      = "train"
MMAP_MODE = 'r'
SAVE_CHKP = True

MODEL_KEYS = [
              'lda', 
              'mlp', 
              'lstm_hcf', 
              'cnn_hcf',
              ]

feature_groups = {
    'WENG':  ['WENG'],
    'RMS':   ['RMS'],
    'HTD':   ['MAV', 'ZC', 'SSC', 'WL'],
    'DFTR':  ['DFTR'],
    'ITD':   ['ISD','COR','MDIFF','MLK'],
    'LS4':   ['LS', 'MFL', 'MSR', 'WAMP'],
    'TDAR':  ['MAV', 'ZC', 'SSC', 'WL', 'AR'],
    'COMB':  ['WL', 'SSC', 'LD', 'AR9'],
    'MSWT':  ['WENG','WV','WWL','WENT'],
    'TDPSD': ['M0','M2','M4','SPARSI','IRF','WLF'],
    'LS9':   ['LS', 'MFL', 'MSR', 'WAMP', 'ZC', 'RMS',
               'IAV', 'DASDV', 'VAR'],
}

# DDP init
dist.init_process_group(backend="nccl")
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = dist.get_world_size()
RANK       = dist.get_rank()
torch.cuda.set_device(LOCAL_RANK)
DEVICE  = f"cuda:{LOCAL_RANK}"
IS_MAIN = (RANK == 0)

SEED = 13 + RANK
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# load raw windows once
TAG = 'raw'
train_windows = np.load(join(PICKLE_PATH, f'train_windows_{TAG}.npy'), mmap_mode=MMAP_MODE)
train_meta    = np.load(join(PICKLE_PATH, f'train_meta_{TAG}.npy'),    allow_pickle=True).item()
val_windows   = np.load(join(PICKLE_PATH, f'val_windows_{TAG}.npy'),   mmap_mode=MMAP_MODE)
val_meta      = np.load(join(PICKLE_PATH, f'val_meta_{TAG}.npy'),      allow_pickle=True).item()
test_windows  = np.load(join(PICKLE_PATH, f'test_windows_{TAG}.npy'),  mmap_mode=MMAP_MODE)
test_meta     = np.load(join(PICKLE_PATH, f'test_meta_{TAG}.npy'),     allow_pickle=True).item()

# main search loop
for feat_key, feat_list in feature_groups.items():
    feat_dic = {}
    for f in feat_list:
        if f in [k.split('_')[0] for k in FEATURE_DIC.keys()]:
            feat_dic[f + '_fs'] = SAMPLING_RATE

    if IS_MAIN:
        print(f"\n{'═'*60}")
        print(f"Feature group: {feat_key}  {feat_list}")
        print(f"{'═'*60}")

    # extract features
    # full window: (N, F) for LDA and MLP
    tr_full  = extract_full(train_windows, feat_list, feat_dic)
    va_full  = extract_full(val_windows,   feat_list, feat_dic)
    te_full  = extract_full(test_windows,  feat_list, feat_dic)
    n_feat   = tr_full.shape[1]   # CH x n_feats_per_channel

    # sub-windowed: (N, 4, F) for LSTM and CNN_Feat
    tr_sub   = extract_sub(train_windows, feat_list, feat_dic)
    va_sub   = extract_sub(val_windows,   feat_list, feat_dic)
    te_sub   = extract_sub(test_windows,  feat_list, feat_dic)
    n_feat_sub = tr_sub.shape[-1]  # F per sub-window

    # normalize - fit on train only
    tr_full, va_full, te_full = normalize_features(tr_full, va_full, te_full)
    tr_sub,  va_sub,  te_sub  = normalize_features(tr_sub,  va_sub,  te_sub)

    # class weights
    weights = None   # data already balanced; no reweighting needed

    for model_key in MODEL_KEYS:
        name = f"{feat_key}_{model_key}"

        # LDA
        if model_key == 'lda':
            if IS_MAIN:
                clf = LinearDiscriminantAnalysis()
                clf.fit(tr_full, train_meta['classes'])
                eval_test_lda(model=clf,
                            X={feat_key: te_full},
                            metas={feat_key: test_meta},
                            name=name)
            dist.barrier() 
            continue

        if model_key == 'mlp':
            model     = MLP(n_features=n_feat)
            tr_data, va_data, te_data = tr_full, va_full, te_full

        elif model_key == 'lstm_hcf':
            model     = LSTM_HCF(n_features=n_feat_sub, n_sub=N_SUB)
            tr_data, va_data, te_data = tr_sub, va_sub, te_sub

        elif model_key == 'cnn_hcf':
            # CNN over sub-windowed features
            # input: (B, n_feat_sub, N_SUB) - features as channels, sub-windows as time
            model     = CNN_HCF(n_feat=n_feat_sub, n_sub=N_SUB)
            # CNN_Feat expects channel-first: transpose (N, 4, F) -> (N, F, 4)
            tr_data   = tr_sub.transpose(0, 2, 1)
            va_data   = va_sub.transpose(0, 2, 1)
            te_data   = te_sub.transpose(0, 2, 1)

        # DDP setup 
        model = model.to(DEVICE)
        model = DDP(model, device_ids=[LOCAL_RANK], find_unused_parameters=False)

        if IS_MAIN:
            print(f"\n  [{name}] params: {count_params(model.module):,}")

        train_loader = create_loader_ddp(tr_data, train_meta['classes'],
                                        train_meta['subjects'],
                                        batch=BATCH_SIZE,
                                        rank=RANK, world_size=WORLD_SIZE,
                                        shuffle=True)
        val_loader   = create_loader_ddp(va_data, val_meta['classes'],
                                        val_meta['subjects'],
                                        batch=BATCH_SIZE,
                                        rank=RANK, world_size=WORLD_SIZE,
                                        shuffle=False)
        if IS_MAIN:
            test_loader = create_loader(te_data, test_meta['classes'],
                                        test_meta['subjects'],
                                        batch=BATCH_SIZE, shuffle=False)

        if MODE == "train":
            train_ddp(model=model, name=name,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    loss_fn=BaseLoss(weight=weights), 
                    return_emb=False, return_logits=True,
                    save_chkp=SAVE_CHKP,
                    rank=RANK, world_size=WORLD_SIZE)
        else:
            map_loc = {"cuda:0": f"cuda:{LOCAL_RANK}"}
            model.module.load_state_dict(
                torch.load(join(CHECKPOINT_PATH, name, f"{name}.pt"),
                        map_location=map_loc)['model_state_dict'])

        if IS_MAIN:
            eval_test(model=model.module, name=name,
                    loaders={feat_key: test_loader},
                    metas={feat_key: test_meta})

        # clean up between iterations
        del model, train_loader, val_loader
        del tr_data, va_data, te_data
        if IS_MAIN:
            del test_loader
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier() 

dist.destroy_process_group()