import warnings, sys, os, gc
from os.path import join
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"

import torch; print(torch.cuda.is_available())

import numpy as np
import random
from filelock import FileLock
from sklearn.utils.class_weight import compute_class_weight

from utils import *
from models import *


SEED = 13; random.seed(SEED); np.random.seed(SEED)
GENERATOR = torch.manual_seed(SEED)
MMAP_MODE = 'r'; SAVE_CHKP = False; VERBOSE=False

TAG = sys.argv[2] if len(sys.argv) > 2 else "raw"
RUNS = sys.argv[3].split(',') if len(sys.argv) > 3 else 15
RUNS = list(map(int, RUNS))


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
val_loader = create_loader(val_windows, val_meta['classes'], val_meta['subjects'], 
                            batch=BATCH_SIZE, shuffle=False)
test_loader = create_loader(test_windows_raw, test_meta_raw['classes'], test_meta_raw['subjects'], 
                            batch=BATCH_SIZE, shuffle=False)

base_ids = train_meta['subjects']
results = {}
_name = f"inc_mhcnn_raw_base"

for i in RUNS:

    if i == 0 or i == 1:
        fname = f"{CHECKPOINT_PATH}/within_mhcnn_raw_base/results-15.npy"
        res = np.load(fname, allow_pickle=True)
        df = pd.DataFrame(res.tolist())
        idx = slice(0, 306)
        acc_mean = df['acc_mean'][idx]
        act_mean = df['act_acc_mean'][idx]
        bal_mean = df['bal_acc_mean'][idx]
        ranked_ids = list(np.argsort(bal_mean))
        if i == 1:
            ranked_ids = list(reversed(ranked_ids))
        unique_ids = np.arange(306)
        mapping = dict(zip(unique_ids, ranked_ids))
        train_meta['subjects'] = np.vectorize(mapping.get)(base_ids.copy())

    if i > 2:
        rng = np.random.default_rng(SEED + i)
        unique_ids = np.arange(306)
        shuffled_ids = rng.permutation(unique_ids)
        mapping = dict(zip(unique_ids, shuffled_ids))
        train_meta['subjects'] = np.vectorize(mapping.get)(base_ids.copy())

    for s in [
              1, 2, 4, 8, 16, 
              32, 64, 128, 
              196, 306,
              ]:
        for r in [
                  1, 2, 4, 8, 16, 
                  24, 32, 40, 50,
                  ]:
            print("S, R:", s, r)
            NAME = _name + f"_seed{i}_s{s}_r{r}"
            indx = (np.isin(train_meta['subjects'], np.arange(s)) & 
                    np.isin(train_meta['reps'], np.arange(r)))
            X = train_windows[indx]
            y = train_meta['classes'][indx]
            ys = train_meta['subjects'][indx]

            _BATCH_SIZE = min(len(X) // 50, BATCH_SIZE)
            _BATCH_SIZE = max(_BATCH_SIZE, 64)

            train_loader = create_loader(X, y, ys,
                batch=_BATCH_SIZE, shuffle=True)

            weights = torch.tensor(compute_class_weight('balanced', 
                classes=np.arange(CLASSES), 
                y=y),
                dtype=torch.float32,
                device=DEVICE)
            weights = None if TAG == 'raw' else weights         # raw is already ~balanced
                
            model = MHCNN()
            train(model=model, name=NAME, 
                train_loader=train_loader,
                val_loader=val_loader, 
                loss_fn=BaseLoss(weight=weights),
                save_chkp=False, verbose=VERBOSE)

            result = eval_test(model=model, name=NAME, save=False,
                    loaders={'raw': test_loader},
                    csv_path=f"{FIGURE_PATH}/inc_{i}.csv",
                    metas={'raw': test_meta_raw})

            print(f"{NAME}: {result['raw']['acc_mean']:2f}")
            results[NAME] = result

            del train_loader, model
            torch.cuda.empty_cache()
            gc.collect()

    # os.makedirs(f"{CHECKPOINT_PATH}/{_name}/", exist_ok=True)
    # np.save(f"{CHECKPOINT_PATH}/{_name}/{_name}_seed{i}_results.npy", results)