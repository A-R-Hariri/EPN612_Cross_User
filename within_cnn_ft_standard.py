import warnings, sys, os, gc
from os.path import join
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"

CPUS = sys.argv[2].split(',') if len(sys.argv) > 2 else ['0', '64']
CPUS = list(map(int, CPUS))
NUM_CPUS = abs(CPUS[1] - CPUS[0])
os.sched_setaffinity(0, set(range(CPUS[0], CPUS[1])))
os.environ["OMP_NUM_THREADS"] = str(NUM_CPUS)
os.environ["MKL_NUM_THREADS"] = str(NUM_CPUS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_CPUS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_CPUS)
os.environ["OMP_DYNAMIC"] = "FALSE"

import torch; print(torch.cuda.is_available())
torch.set_num_threads(NUM_CPUS)
torch.set_num_interop_threads(max(4, NUM_CPUS // 8))

import libemg
from libemg.datasets import get_dataset_list
from libemg.feature_extractor import FeatureExtractor

import numpy as np, pandas as pd
import random, copy, time
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from utils import *
from models import *


WORKERS = 0; PRE_FETCH = 2; VERBOSE=False; BATCH_SIZE=128
PRESIST_WORKER = False; PIN_MEMORY = True

SEED = 13; random.seed(SEED); np.random.seed(SEED)
GENERATOR = torch.manual_seed(SEED)
MMAP_MODE = 'r'; SAVE_CHKP = False


# ======== LOAD DATA ========
train_data_standard = np.load(join(PICKLE_PATH, 'train_data_standard.npy'), allow_pickle=True).item()
val_data_standard = np.load(join(PICKLE_PATH, 'val_data_standard.npy'), allow_pickle=True).item()
test_data_standard = np.load(join(PICKLE_PATH, 'test_data_standard.npy'), allow_pickle=True).item()
train_data_standard.extra_attributes.remove('base_class')
val_data_standard.extra_attributes.remove('base_class')
test_data_standard.extra_attributes.remove('base_class')

train_windows_standard = np.load(join(PICKLE_PATH, 'train_windows_standard.npy'), mmap_mode=MMAP_MODE)
train_meta_standard = np.load(join(PICKLE_PATH, 'train_meta_standard.npy'), allow_pickle=True).item()
val_windows_standard = np.load(join(PICKLE_PATH, 'val_windows_standard.npy'), mmap_mode=MMAP_MODE)
val_meta_standard = np.load(join(PICKLE_PATH, 'val_meta_standard.npy'), allow_pickle=True).item()
test_windows_standard = np.load(join(PICKLE_PATH, 'test_windows_standard.npy'), mmap_mode=MMAP_MODE)
test_meta_standard = np.load(join(PICKLE_PATH, 'test_meta_standard.npy'), allow_pickle=True).item()


REPS = sys.argv[2].split(',') if len(sys.argv) > 2 else 15
REPS = list(map(int, REPS))

for rep in REPS:

    # Within
    NAME = f'cnn_standard_within_ft_{rep}'
    results = []

    ranges = [(0, 306), (306, 332), (332, 612)]
    data_list = [train_data_standard, val_data_standard, test_data_standard]

    for d, r in enumerate(ranges):
        for i in range(*r):
            print(i)

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
                                        batch=BATCH_SIZE, shuffle=True, 
                                        workers=WORKERS, persistent_workers=PRESIST_WORKER)
            val_loader = create_loader(val_windows, val_meta['classes'], 
                                        batch=BATCH_SIZE, shuffle=False, 
                                        workers=WORKERS, persistent_workers=PRESIST_WORKER)
            test_loader = create_loader(test_windows, test_meta['classes'], 
                                        batch=BATCH_SIZE, shuffle=False, 
                                        workers=WORKERS, persistent_workers=PRESIST_WORKER)

            model = CNN()
            model.load_state_dict(torch.load(join(CHECKPOINT_PATH, "cnn_standard", "cnn_standard.pt")))
            weights = torch.tensor(compute_class_weight('balanced', 
                                        classes=np.arange(CLASSES), 
                                            y=train_meta['classes']),
                                            dtype=torch.float32,
                                            device=DEVICE)
            train(model=model, name=NAME, 
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=nn.CrossEntropyLoss(weight=weights),
                save_chkp=SAVE_CHKP, verbose=VERBOSE)
            _result = eval_within(model=model,
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


    # Within Eq
    NAME = f'cnn_standard_within_ft_eq_{rep}'
    results = []

    ranges = [(0, 306), (306, 332), (332, 612)]
    data_list = [train_data_standard, val_data_standard, test_data_standard]

    for d, r in enumerate(ranges):
        for i in range(*r):
            print(i)

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
                                        batch=BATCH_SIZE, shuffle=True, 
                                        workers=WORKERS, persistent_workers=PRESIST_WORKER)
            val_loader = create_loader(val_windows, val_meta['classes'], 
                                        batch=BATCH_SIZE, shuffle=False, 
                                        workers=WORKERS, persistent_workers=PRESIST_WORKER)
            test_loader = create_loader(test_windows, test_meta['classes'], 
                                        batch=BATCH_SIZE, shuffle=False, 
                                        workers=WORKERS, persistent_workers=PRESIST_WORKER)

            model = CNN()
            model.load_state_dict(torch.load(join(CHECKPOINT_PATH, "cnn_standard_eq", "cnn_standard_eq.pt")))
            weights = torch.tensor(compute_class_weight('balanced', 
                                        classes=np.arange(CLASSES), 
                                            y=train_meta['classes']),
                                            dtype=torch.float32,
                                            device=DEVICE)
            train(model=model, name=NAME, 
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=EqLoss(weight=weights),
                save_chkp=SAVE_CHKP, verbose=VERBOSE)
            _result = eval_within(model=model,
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


    # Within CVaR
    NAME = f'cnn_standard_within_cvar_ft_{rep}'
    results = []

    ranges = [(0, 306), (306, 332), (332, 612)]
    data_list = [train_data_standard, val_data_standard, test_data_standard]

    for d, r in enumerate(ranges):
        for i in range(*r):
            print(i)

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
                                        batch=BATCH_SIZE, shuffle=True, 
                                        workers=WORKERS, persistent_workers=PRESIST_WORKER)
            val_loader = create_loader(val_windows, val_meta['classes'], 
                                        batch=BATCH_SIZE, shuffle=False, 
                                        workers=WORKERS, persistent_workers=PRESIST_WORKER)
            test_loader = create_loader(test_windows, test_meta['classes'], 
                                        batch=BATCH_SIZE, shuffle=False, 
                                        workers=WORKERS, persistent_workers=PRESIST_WORKER)

            model = CNN()
            model.load_state_dict(torch.load(join(CHECKPOINT_PATH, "cnn_standard_cvar", "cnn_standard_cvar.pt")))
            weights = torch.tensor(compute_class_weight('balanced', 
                                        classes=np.arange(CLASSES), 
                                            y=train_meta['classes']),
                                            dtype=torch.float32,
                                            device=DEVICE)
            train(model=model, name=NAME, 
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=CVaRLoss(weight=weights),
                save_chkp=SAVE_CHKP, verbose=VERBOSE)
            _result = eval_within(model=model,
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


    # Within Rest
    NAME = f'cnn_standard_within_rest_ft_{rep}'
    results = []

    ranges = [(0, 306), (306, 332), (332, 612)]
    data_list = [train_data_standard, val_data_standard, test_data_standard]

    for d, r in enumerate(ranges):
        for i in range(*r):
            print(i)

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
                                        batch=BATCH_SIZE, shuffle=True, 
                                        workers=WORKERS, persistent_workers=PRESIST_WORKER)
            val_loader = create_loader(val_windows, val_meta['classes'], 
                                        batch=BATCH_SIZE, shuffle=False, 
                                        workers=WORKERS, persistent_workers=PRESIST_WORKER)
            test_loader = create_loader(test_windows, test_meta['classes'], 
                                        batch=BATCH_SIZE, shuffle=False, 
                                        workers=WORKERS, persistent_workers=PRESIST_WORKER)

            model = CNN()
            model.load_state_dict(torch.load(join(CHECKPOINT_PATH, "cnn_standard_rest", "cnn_standard_rest.pt")))
            weights = torch.tensor(compute_class_weight('balanced', 
                                        classes=np.arange(CLASSES), 
                                            y=train_meta['classes']),
                                            dtype=torch.float32,
                                            device=DEVICE)
            train(model=model, name=NAME, 
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=RestLoss(1.0, 1.0, weight=weights),
                save_chkp=SAVE_CHKP, verbose=VERBOSE)
            _result = eval_within(model=model,
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