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


# ======== LOAD DATA ========
train_windows = np.load(join(PICKLE_PATH, 'train_windows.npy'), mmap_mode=MMAP_MODE)
train_meta = np.load(join(PICKLE_PATH, 'train_meta.npy'), allow_pickle=True).item()
val_windows = np.load(join(PICKLE_PATH, 'val_windows.npy'), mmap_mode=MMAP_MODE)
val_meta = np.load(join(PICKLE_PATH, 'val_meta.npy'), allow_pickle=True).item()
test_windows = np.load(join(PICKLE_PATH, 'test_windows.npy'), mmap_mode=MMAP_MODE)
test_meta = np.load(join(PICKLE_PATH, 'test_meta.npy'), allow_pickle=True).item()

test_windows_segmented = np.load(join(PICKLE_PATH, 'test_windows_segmented.npy'), mmap_mode=MMAP_MODE)
test_meta_segmented = np.load(join(PICKLE_PATH, 'test_meta_segmented.npy'), allow_pickle=True).item()

test_windows_relabeled = np.load(join(PICKLE_PATH, 'test_windows_relabeled.npy'), mmap_mode=MMAP_MODE)
test_meta_relabeled = np.load(join(PICKLE_PATH, 'test_meta_relabeled.npy'), allow_pickle=True).item()

test_windows_standard = np.load(join(PICKLE_PATH, 'test_windows_standard.npy'), mmap_mode=MMAP_MODE)
test_meta_standard = np.load(join(PICKLE_PATH, 'test_meta_standard.npy'), allow_pickle=True).item()


# ======== PIPELINE ========
train_loader = create_loader(train_windows, train_meta['classes'], 
                            batch=BATCH_SIZE, shuffle=True)
train_loader_sbj = create_loader_sbj(train_windows, train_meta['classes'], 
                                     train_meta['subjects'],
                                     batch=BATCH_SIZE, shuffle=True)

val_loader = create_loader(val_windows, val_meta['classes'], 
                            batch=BATCH_SIZE, shuffle=True)
val_loader_sbj = create_loader_sbj(val_windows, val_meta['classes'], 
                                   val_meta['subjects'],
                                   batch=BATCH_SIZE, shuffle=True)

test_loader = create_loader(test_windows, test_meta['classes'], 
                            batch=BATCH_SIZE, shuffle=False)
test_loader_sbj = create_loader_sbj(test_windows, test_meta['classes'], 
                                   test_meta['subjects'],
                                   batch=BATCH_SIZE, shuffle=False)


test_loader_segmented = create_loader(test_windows_segmented, 
                            test_meta_segmented['classes'], 
                            batch=BATCH_SIZE, shuffle=False)
test_loader_segmented_sbj = create_loader_sbj(test_windows_segmented, 
                            test_meta_segmented['classes'], 
                            test_meta_segmented['subjects'], 
                            batch=BATCH_SIZE, shuffle=False)


test_loader_relabeled = create_loader(test_windows_relabeled, 
                            test_meta_relabeled['classes'],
                            batch=BATCH_SIZE, shuffle=False)
test_loader_relabeled_sbj = create_loader_sbj(test_windows_relabeled, 
                            test_meta_relabeled['classes'],
                            test_meta_relabeled['subjects'],
                            batch=BATCH_SIZE, shuffle=False)


test_loader_standard = create_loader(test_windows_standard, 
                            test_meta_standard['classes'],
                            batch=BATCH_SIZE, shuffle=False)
test_loader_standard_sbj = create_loader_sbj(test_windows_standard, 
                            test_meta_standard['classes'],
                            test_meta_standard['subjects'],
                            batch=BATCH_SIZE, shuffle=False)

weights = torch.tensor(compute_class_weight('balanced', 
                               classes=np.arange(CLASSES), 
                                y=train_meta['classes']),
                                dtype=torch.float32,
                                device=DEVICE)


MODE = sys.argv[2] if len(sys.argv) > 2 else "train"


NAME = "cnn_raw"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train(model=model, name=NAME, 
          train_loader=train_loader,
          val_loader=val_loader, 
          loss_fn=nn.CrossEntropyLoss(weight=weights),
          save_chkp=SAVE_CHKP)
    torch.save(model.state_dict(), join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
else:
    model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader},
           metas={'raw': test_meta})


NAME = "cnn_raw_std"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train(model=model, name=NAME, 
          train_loader=train_loader,
          val_loader=val_loader, 
          loss_fn=STDLoss(),
          save_chkp=SAVE_CHKP)
    torch.save(model.state_dict(), join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
else:
    model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader},
           metas={'raw': test_meta})


NAME = "cnn_raw_cvar"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train(model=model, name=NAME, 
          train_loader=train_loader,
          val_loader=val_loader, 
          loss_fn=CVaRLoss(weight=weights),
          save_chkp=SAVE_CHKP)
    torch.save(model.state_dict(), join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
else:
    model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader},
           metas={'raw': test_meta})


NAME = "cnn_raw_rest"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train(model=model, name=NAME, 
          train_loader=train_loader,
          val_loader=val_loader, 
          loss_fn=RestLoss(weight=weights),
          save_chkp=SAVE_CHKP)
    torch.save(model.state_dict(), join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
else:
    model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader},
           metas={'raw': test_meta})


NAME = "cnn_raw_act"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train(model=model, name=NAME, 
          train_loader=train_loader,
          val_loader=val_loader, 
          loss_fn=ActiveLoss(weight=weights),
          save_chkp=SAVE_CHKP)
    torch.save(model.state_dict(), join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
else:
    model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader},
           metas={'raw': test_meta})


NAME = "cnn_raw_grl"
model = CNN_GRL()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train_grl(model=model, name=NAME, 
          train_loader=train_loader_sbj,
          val_loader=val_loader, 
          loss_fn=nn.CrossEntropyLoss(weight=weights),
          loss_fn_sbj=nn.CrossEntropyLoss(),
          save_chkp=SAVE_CHKP)
    torch.save(model.state_dict(), join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
else:
    model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader},
           metas={'raw': test_meta})


NAME = "cnn_raw_sbj"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train_sbj(model=model, name=NAME, 
          train_loader=train_loader_sbj,
          val_loader=val_loader_sbj, 
          loss_fn=PerSubjectLoss(weight=weights),
          save_chkp=SAVE_CHKP)
    torch.save(model.state_dict(), join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
else:
    model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader},
           metas={'raw': test_meta})


NAME = "cnn_raw_proto"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train(model=model, name=NAME, 
          train_loader=train_loader,
          val_loader=val_loader, 
          loss_fn=PrototypeLoss(weight=weights),
          return_emb=True, return_logits=True,
          save_chkp=SAVE_CHKP)
    torch.save(model.state_dict(), join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
else:
    model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader},
           metas={'raw': test_meta})


NAME = "cnn_raw_1va"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train(model=model, name=NAME, 
          train_loader=train_loader,
          val_loader=val_loader, 
          loss_fn=OneVsAllLoss(weight=weights),
          return_emb=True, return_logits=True,
          save_chkp=SAVE_CHKP)
    torch.save(model.state_dict(), join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
else:
    model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader},
           metas={'raw': test_meta})


c = np.unique(train_meta['classes'], return_counts=True)[1]
indx_t = np.random.random(len(train_meta['classes'])) < c[np.argsort(c)[:-1]].mean() / c.max()
train_loader = create_loader(train_windows[indx_t], train_meta['classes'][indx_t], 
                            batch=BATCH_SIZE, shuffle=True)
train_loader_sbj = create_loader_sbj(train_windows[indx_t], train_meta['classes'][indx_t], 
                                    train_meta['subjects'][indx_t],
                                    batch=BATCH_SIZE, shuffle=True)

c = np.unique(val_meta['classes'], return_counts=True)[1]
indx_v = np.random.random(len(val_meta['classes'])) < c[np.argsort(c)[:-1]].mean() / c.max()
val_loader = create_loader(val_windows[indx_v], val_meta['classes'][indx_v], 
                            batch=BATCH_SIZE, shuffle=True)
val_loader_sbj = create_loader_sbj(val_windows[indx_v], val_meta['classes'][indx_v], 
                                val_meta['subjects'][indx_v],
                                batch=BATCH_SIZE, shuffle=True)


NAME = "cnn_raw-bal"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train(model=model, name=NAME, 
        train_loader=train_loader,
        val_loader=val_loader, 
        loss_fn=nn.CrossEntropyLoss(weight=None),
        save_chkp=SAVE_CHKP)
    torch.save(model.state_dict(), join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
else:
    model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader},
           metas={'raw': test_meta})

    
NAME = "cnn_raw_std-bal"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train(model=model, name=NAME, 
        train_loader=train_loader,
        val_loader=val_loader, 
        loss_fn=STDLoss(),
        save_chkp=SAVE_CHKP)
    torch.save(model.state_dict(), join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
else:
    model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader},
           metas={'raw': test_meta})


NAME = "cnn_raw_rest-bal"
model = CNN()
print(model, f"\nParameters count: {count_params(model):,}")
if MODE == "train":
    train(model=model, name=NAME, 
        train_loader=train_loader,
        val_loader=val_loader, 
        loss_fn=RestLoss(weight=None),
        save_chkp=SAVE_CHKP)
    torch.save(model.state_dict(), join(CHECKPOINT_PATH, NAME, f"{NAME}.pt"))
else:
    model.load_state_dict(torch.load(join(CHECKPOINT_PATH, NAME, f"{NAME}.pt")))
eval_test(model=model, name=NAME, 
          loaders={'raw': test_loader},
           metas={'raw': test_meta})
