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
REPS = sys.argv[3].split(',') if len(sys.argv) > 3 else 15
REPS = list(map(int, REPS))
FT = sys.argv[4] == 'ft' if len(sys.argv) > 4 else False

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
        
        NAME = f'within_mhcnn_{TAG}_{l_name}{"-ft" if FT else ""}-{rep}'
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

                model = MHCNN()
                
                if FT:
                    try:
                        model.load_state_dict(torch.load(join
                            (CHECKPOINT_PATH, f'mhcnn_{TAG}_{l_name}', 
                            f'mhcnn_{TAG}_{l_name}.pt'))['model_state_dict'])
                    except:
                        checkpoint = torch.load(join
                            (CHECKPOINT_PATH, f'mhcnn_{TAG}_{l_name}', 
                            f'mhcnn_{TAG}_{l_name}.pt'))['model_state_dict']
                        new_state_dict = {}
                        for k, v in checkpoint.items():
                            new_state_dict[k.replace("module.", "")] = v
                        model.load_state_dict(new_state_dict)

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
        os.makedirs(f"{CHECKPOINT_PATH}/{_name}/", exist_ok=True)
        np.save(f"{CHECKPOINT_PATH}/{_name}/results-{rep}.npy", results)
