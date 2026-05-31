# ══════════════════════════════════════════════
# cross_models.py
# ══════════════════════════════════════════════
import warnings, sys, os
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import random
from os.path import join
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from utils import *
from models import *

# config
MODE      = "train"
MMAP_MODE = 'r'
SAVE_CHKP = True
N_SUB     = 4       # sub-windows for LSTM_SubRMS (4 × 50ms)
SAMPLING_RATE = 200


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

feature_groups = {
    'HTD':   ['MAV', 'ZC', 'SSC', 'WL'],
    'TSTD':  ['MAVFD','DASDV','WAMP','ZC','MFL','SAMPEN',
               'M0','M2','M4','SPARSI','IRF','WLF'],
    'DFTR':  ['DFTR'],
    'ITD':   ['ISD','COR','MDIFF','MLK'],
    'HJORTH':['ACT','MOB','COMP'],
    'LS4':   ['LS', 'MFL', 'MSR', 'WAMP'],
    'LS9':   ['LS', 'MFL', 'MSR', 'WAMP', 'ZC', 'RMS',
               'IAV', 'DASDV', 'VAR'],
    'TDPSD': ['M0','M2','M4','SPARSI','IRF','WLF'],
    'TDAR':  ['MAV', 'ZC', 'SSC', 'WL', 'AR'],
    'COMB':  ['WL', 'SSC', 'LD', 'AR9'],
    'MSWT':  ['WENG','WV','WWL','WENT'],
    'RMS':   ['RMS'],
    'WENG':  ['WENG'],
}
feat_list = feature_groups['WENG'] # Decided based on cross_feats.py
feat_dic = {}
for f in feat_list:
    if f in [k.split('_')[0] for k in FEATURE_DIC.keys()]:
        feat_dic[f + '_fs'] = SAMPLING_RATE

# load raw windows
TAG = 'raw'
train_windows = np.load(join(PICKLE_PATH, f'train_windows_{TAG}.npy'), mmap_mode=MMAP_MODE)
train_meta    = np.load(join(PICKLE_PATH, f'train_meta_{TAG}.npy'),    allow_pickle=True).item()
val_windows   = np.load(join(PICKLE_PATH, f'val_windows_{TAG}.npy'),   mmap_mode=MMAP_MODE)
val_meta      = np.load(join(PICKLE_PATH, f'val_meta_{TAG}.npy'),      allow_pickle=True).item()
test_windows  = np.load(join(PICKLE_PATH, f'test_windows_{TAG}.npy'),  mmap_mode=MMAP_MODE)
test_meta     = np.load(join(PICKLE_PATH, f'test_meta_{TAG}.npy'),     allow_pickle=True).item()

if IS_MAIN:
    print(f"\n{'═'*60}")
    print(f"Feature group: {feat_list}")
    print(f"{'═'*60}")

# extract features 
# full window: (N, F) for LDA and MLP
tr_full  = extract_full(train_windows, feat_list, feat_dic)
va_full  = extract_full(val_windows,   feat_list, feat_dic)
te_full  = extract_full(test_windows,  feat_list, feat_dic)
n_feat   = tr_full.shape[1]   # CH × n_feats_per_channel

# sub-windowed: (N, 4, F) for LSTM and CNN_Feat
tr_sub   = extract_sub(train_windows, feat_list, feat_dic)
va_sub   = extract_sub(val_windows,   feat_list, feat_dic)
te_sub   = extract_sub(test_windows,  feat_list, feat_dic)
n_feat_sub = tr_sub.shape[-1]  # F per sub-window

# normalize — fit on train only
train_hcf, val_hcf, test_hcf = normalize_features(tr_full, va_full, te_full)
train_shcf, val_shcf, test_shcf  = normalize_features(tr_sub,  va_sub,  te_sub)


# class weights (on training labels)
weights = torch.tensor(
    compute_class_weight('balanced',
                         classes=np.arange(CLASSES),
                         y=train_meta['classes']),
    dtype=torch.float32, device=DEVICE)
weights = None       # data already ~balanced; no reweighting needed

# model registry
#   key → (model_instance, train_data, val_data, test_data, name)

REGISTRY = {
    'lda': None,   # handled separately below, sklearn only

    'mlp': dict(
        model      = MLP(n_features=n_feat),
        train_data = train_hcf,       # (N, 8)
        val_data   = val_hcf,
        test_data  = test_hcf,
    ),

    # LSTM on sub-windowed RMS: (B, 4, 8) 
    'lstm_hcf': dict(
        model      = LSTM_HCF(n_feat_sub, n_sub=N_SUB),
        train_data = train_shcf,    # (N, 4, 8)
        val_data   = val_shcf,
        test_data  = test_shcf,
    ),

    # raw EMG: (B, 8, 40)
    'cnn_hcf': dict(
        model      = CNN_HCF(n_feat_sub),
        train_data = train_shcf.transpose(0, 2, 1),    # (N, 8, 4)
        val_data   = val_shcf.transpose(0, 2, 1),
        test_data  = test_shcf.transpose(0, 2, 1),
    ),

    # raw EMG: (B, 8, 40)
    'lstm': dict(
        model      = LSTM(),
        train_data = train_windows,   # (N, 8, 40)
        val_data   = val_windows,
        test_data  = test_windows,
    ),

    'cnn': dict(
        model      = CNN(),
        train_data = train_windows,   # (N, 8, 40)
        val_data   = val_windows,
        test_data  = test_windows,
    ),

    'mhcnn_raw_base': dict(
        model      = MHCNN(),
        train_data = train_windows,   # (N, 8, 40)
        val_data   = val_windows,
        test_data  = test_windows,
    ),
}

for MODEL_KEY in REGISTRY.keys():
    if MODEL_KEY == 'lda':
        if IS_MAIN:
            clf = LinearDiscriminantAnalysis()
            clf.fit(train_hcf, train_meta['classes'])
            eval_test_lda(
                model=clf,
                X={TAG: test_hcf},
                metas={TAG: test_meta},
                name='lda'
            )
        dist.barrier()       
        continue

    # DDP training for nn.Module models
    cfg  = REGISTRY[MODEL_KEY]
    NAME = MODEL_KEY

    model = cfg['model'].to(DEVICE)
    model = DDP(model, device_ids=[LOCAL_RANK], find_unused_parameters=False)

    if IS_MAIN:
        print(model.module, f"\nParameters: {count_params(model.module):,}")

    train_loader = create_loader_ddp(cfg['train_data'], train_meta['classes'],
                                    train_meta['subjects'],
                                    batch=BATCH_SIZE,
                                    rank=RANK, world_size=WORLD_SIZE,
                                    shuffle=True)
    val_loader   = create_loader_ddp(cfg['val_data'], val_meta['classes'],
                                    val_meta['subjects'],
                                    batch=BATCH_SIZE,
                                    rank=RANK, world_size=WORLD_SIZE,
                                    shuffle=False)
    if IS_MAIN:
        test_loader = create_loader(cfg['test_data'], test_meta['classes'],
                                    test_meta['subjects'],
                                    batch=BATCH_SIZE, shuffle=False)

    if MODE == "train":
        train_ddp(model=model, name=NAME,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=BaseLoss(weight=weights),
                return_emb=False, return_logits=True,
                save_chkp=SAVE_CHKP,
                rank=RANK, world_size=WORLD_SIZE)
    else:
        map_loc = {"cuda:0": f"cuda:{LOCAL_RANK}"}
        model.module.load_state_dict(
            torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"),
                    map_location=map_loc)['model_state_dict'])

    if IS_MAIN:
        eval_test(model=model.module, name=NAME,
                loaders={TAG: test_loader},
                metas={TAG: test_meta})

    del model, train_loader, val_loader
    if IS_MAIN:
        del test_loader
    torch.cuda.empty_cache()
    dist.barrier()          # all ranks sync before next model

dist.destroy_process_group()