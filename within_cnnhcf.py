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


WORKERS = 0; PRE_FETCH = 2; VERBOSE=False
BATCH_SIZE=64; PATIENCE = 5; LR_PATIENCE = 3
PRESIST_WORKER = False; PIN_MEMORY = True

SEED = 13; random.seed(SEED); np.random.seed(SEED)
GENERATOR = torch.manual_seed(SEED)
MMAP_MODE = 'r'; SAVE_CHKP = False

TAG = sys.argv[2] if len(sys.argv) > 2 else "raw"
REPS = sys.argv[3].split(',') if len(sys.argv) > 3 else [15]
REPS = list(map(int, REPS))
FT = sys.argv[4] == 'ft' if len(sys.argv) > 4 else False


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


# ======== LOAD DATA ========
train_data = np.load(join(PICKLE_PATH, f'train_data_{TAG}.npy'), allow_pickle=True).item()
val_data = np.load(join(PICKLE_PATH, f'val_data_{TAG}.npy'), allow_pickle=True).item()
test_data = np.load(join(PICKLE_PATH, f'test_data_{TAG}.npy'), allow_pickle=True).item()


# ======== PIPELINE ========
for rep in REPS:

    # -------- BASE --------
    for _loss, l_name in [
                        (BaseLoss, 'base'),
                        # (RestLoss, 'rest'),
                        # (PrototypeLoss, 'proto'),
                        # (OneVsAllLoss, '1va'),
                        # (AngularLoss, 'ang'),
                        ]:
        
        NAME = f'within_cnnhcf_{TAG}_{l_name}{"-ft" if FT else ""}-{rep}'
        results = []

        ranges = [(0, 306), (306, 332), (332, 612)]
        data_list = [train_data, val_data, test_data]

        return_emb = l_name in ['proto', '1va', 'ang']

        for d, r in enumerate(ranges):
            for i in range(*r):
                print(i, NAME)

                data_s = data_list[d].isolate_data("subjects", [i], fast=True)

                data = data_s.isolate_data("reps", list(range(rep)), fast=True)
                train_windows, train_meta = data.parse_windows(SEQ, INC)

                data = data_s.isolate_data("reps", list(range(15, 20)), fast=True)
                val_windows, val_meta = data.parse_windows(SEQ, INC)

                data = data_s.isolate_data("reps", list(range(20, 25)), fast=True)
                test_windows, test_meta = data.parse_windows(SEQ, INC)

                train_windows   = extract_sub(train_windows, feat_list, feat_dic)
                val_windows   = extract_sub(val_windows,   feat_list, feat_dic)
                test_windows   = extract_sub(test_windows,  feat_list, feat_dic)
                n_feat_sub = train_windows.shape[-1]  # F per sub-window

                train_windows, val_windows, test_windows  = normalize_features(
                                        train_windows,  val_windows,  test_windows)
                
                train_windows = train_windows.transpose(0, 2, 1)
                val_windows = val_windows.transpose(0, 2, 1)
                test_windows = test_windows.transpose(0, 2, 1)

                weights = torch.tensor(compute_class_weight('balanced', 
                                            classes=np.arange(CLASSES), 
                                                y=train_meta['classes']),
                                                dtype=torch.float32,
                                                device=DEVICE)

                train_loader = create_loader(train_windows, train_meta['classes'], 
                                            train_meta['subjects'], 
                                            batch=BATCH_SIZE, shuffle=True, 
                                            workers=WORKERS, persistent_workers=PRESIST_WORKER)
                val_loader = create_loader(val_windows, val_meta['classes'], 
                                            val_meta['subjects'], 
                                            batch=BATCH_SIZE, shuffle=False, 
                                            workers=WORKERS, persistent_workers=PRESIST_WORKER)
                test_loader = create_loader(test_windows, test_meta['classes'], 
                                            test_meta['subjects'], 
                                            batch=BATCH_SIZE, shuffle=False, 
                                            workers=WORKERS, persistent_workers=PRESIST_WORKER)

                model = CNN_HCF(n_feat_sub)

                if FT:
                    model.load_state_dict(torch.load(join
                        (CHECKPOINT_PATH, f'cnn_hcf', 
                        f'cnn_hcf.pt'))['model_state_dict'])

                weights = torch.tensor(compute_class_weight('balanced', 
                                            classes=np.arange(CLASSES), 
                                                y=train_meta['classes']),
                                                dtype=torch.float32,
                                                device=DEVICE)
                train(model=model, name=NAME, 
                    train_loader=train_loader,
                    val_loader=val_loader,
                    return_emb=return_emb,
                    return_logits=return_emb,
                    loss_fn=_loss(weight=weights),
                    save_chkp=SAVE_CHKP, verbose=VERBOSE)
                _result = eval_within(model=model,
                                    loader=test_loader,
                                    meta=test_meta)
                results.append(_result)
                print(_result['acc_mean'])

                del train_loader, val_loader, test_loader, model
                torch.cuda.empty_cache()
                gc.collect()

        _name = NAME.split('-')[0]
        _name = f'{_name}{"-ft" if FT else ""}'
        os.makedirs(f"{CHECKPOINT_PATH}/{_name}/", exist_ok=True)
        np.save(f"{CHECKPOINT_PATH}/{_name}/results-{rep}.npy", results)
