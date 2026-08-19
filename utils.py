import os, copy, time, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor

import torch;
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn; import torch.nn.functional as F
from torch.optim import Adam
from torch.amp import GradScaler, autocast
from torch.utils.data import (DataLoader, TensorDataset, Sampler)
from sklearn.preprocessing import StandardScaler
from libemg.feature_extractor import FeatureExtractor
from torch.nn.utils import clip_grad_norm_


def is_notebook():
    try:
        from IPython import get_ipython; shell = get_ipython()
        if shell is None: return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except: return False

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


DTYPE = np.float32
PICKLE_PATH = 'pickles'; CHECKPOINT_PATH = 'checkpoints'; 
FIGURE_PATH = 'figures'; RESULTS_PATH = f"{FIGURE_PATH}/results.csv"
UPDATE_EVERY = 50; PRESIST_WORKER = False; PIN_MEMORY = True

SEQ = 40; INC = 5; CH = 8; CLASSES = 5; VAL_CUTOFF = 332
WORKERS = 4; PRE_FETCH = 2; VERBOSE=True; DEVICE = 'cuda'

_GESTURE_LABELS = {0: "NM", 1: "HC", 2: "FX", 3: "EX", 4: "HO"}

EPOCHS = 300; BATCH_SIZE = 2048; DROPOUT = 0.2; PATIENCE = 15
LR_FACTOR = 0.6; LR_PATIENCE = 7; LR_INIT = 5e-4; LR_MIN = 1e-6

RN_PRIOR_WEIGHT = 0
N_SUB     = 4       # 4 sub-windows × 10 samples = 40 samples total

N_SUBJECTS = 306; MARGIN = 0.5; W_HARD = 1.0; W_SOFT = 0.0
ALPHA_START = 0.01; ALPHA_END = 0.25; WARMUP = 25
TAU = float('inf')

SAMPLING_RATE = 200
FEATURE_DIC = {
               'WENG_fs': SAMPLING_RATE,
               'DFTR_fs': SAMPLING_RATE,
               'MDF_fs': SAMPLING_RATE,
               'MNF_fs': SAMPLING_RATE,
               'SM_fs': SAMPLING_RATE,
               'WV_fs': SAMPLING_RATE,
               'WENT_fs': SAMPLING_RATE,
               }


# ======== MODELS, TRAINING & DATASETS ========
def count_params(m): 
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ======== DATA LOADER ========
def create_loader(x, y, s, batch=BATCH_SIZE, shuffle=False, 
                  workers=WORKERS, prefetch_factor=PRE_FETCH,
                  persistent_workers=PRESIST_WORKER,
                  pin_memory=PIN_MEMORY):
    return DataLoader(
            TensorDataset(torch.from_numpy(x), 
                            torch.from_numpy(y),
                            torch.from_numpy(s)),
            batch_size=batch,
            shuffle=shuffle,
            num_workers=workers,
            prefetch_factor=prefetch_factor if workers > 0 else None,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            drop_last=False)


# -------- TRIPLET SAMPLER --------
class TripletBatchSampler(Sampler):
    def __init__(
        self,
        labels,
        subjects,
        batch_size,
        n_classes,
        n_subjects,
        *,
        seed_offset=0,
        reuse_mode="random_start",  # "random_start" or "replacement"
):
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.subjects_raw = torch.as_tensor(subjects, dtype=torch.long)

        self.n_classes = int(n_classes)
        self.n_subjects = int(n_subjects)
        if self.n_subjects < 2:
            raise ValueError("n_subjects must be >= 2.")
        if self.n_classes < 2:
            raise ValueError("n_classes must be >= 2.")

        # n_samples inferred from nominal batch_size
        self.n_samples = max(3, int(batch_size) // (self.n_classes * self.n_subjects))
        self.batch_size = self.n_classes * self.n_subjects * self.n_samples

        # Remap subjects to 0..S-1 (handles train 0..305 and test 305..611)
        uniq_subj = torch.unique(self.subjects_raw).sort()[0]
        self.S = int(uniq_subj.numel())
        if self.S < self.n_subjects:
            raise ValueError("Not enough subjects for n_subjects.")

        # Build mapping: raw_id -> [0..S-1]
        # Use searchsorted because uniq_subj is sorted and subjects are guaranteed from it
        self.subjects = torch.searchsorted(uniq_subj, self.subjects_raw)

        # Validate labels range cheaply
        if self.labels.min().item() < 0 or self.labels.max().item() >= self.n_classes:
            raise ValueError("labels must be in [0..n_classes-1].")

        # Cell id in [0..(n_classes*S-1)]
        self.cell_id = self.labels * self.S + self.subjects
        K = self.n_classes * self.S

        # Sort indices by cell to get contiguous blocks per cell
        self.order = torch.argsort(self.cell_id)
        cell_sorted = self.cell_id[self.order]

        # counts and starts per cell
        self.counts = torch.bincount(cell_sorted, minlength=K)
        self.starts = torch.zeros(K + 1, dtype=torch.long)
        self.starts[1:] = torch.cumsum(self.counts, dim=0)

        self.cursor = torch.zeros(K, dtype=torch.long)

        # Epoch length based on total N (large epochs, no discard)
        self.length = int(self.labels.numel()) // int(self.batch_size)
        if self.length <= 0:
            raise ValueError("Dataset too small for chosen batch structure.")

        self.epoch = 0
        self.seed_offset = int(seed_offset)
        self.reuse_mode = reuse_mode

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.length

    def __iter__(self):
            g = torch.Generator()
            g.manual_seed(torch.initial_seed() + self.seed_offset + self.epoch)

            subj_perm = torch.randperm(self.S, generator=g)
            subj_ptr = 0

            def take_from_cell(cell: int) -> torch.Tensor:
                cnt = int(self.counts[cell].item())
                if cnt <= 0:
                    ridx = torch.randint(0, self.order.numel(), (self.n_samples,), generator=g)
                    return self.order[ridx]

                p = int(self.cursor[cell].item()) 

                # sequential without replacement
                if p + self.n_samples <= cnt:
                    lo = int(self.starts[cell].item()) + p
                    hi = lo + self.n_samples
                    out = self.order[lo:hi]
                    self.cursor[cell] = p + self.n_samples 
                    return out

                # exhausted: reuse (no toss)
                if self.reuse_mode == "replacement":
                    lo0 = int(self.starts[cell].item())
                    ridx = torch.randint(0, cnt, (self.n_samples,), generator=g)
                    return self.order[lo0 + ridx]

                # default: random_start contiguous block + pad if needed
                if cnt > self.n_samples:
                    p0 = int(torch.randint(0, cnt - self.n_samples + 1, (1,), generator=g).item())
                else:
                    p0 = 0

                lo = int(self.starts[cell].item()) + p0
                hi = lo + min(self.n_samples, cnt)
                out = self.order[lo:hi]

                if out.numel() < self.n_samples:
                    lo0 = int(self.starts[cell].item())
                    ridx = torch.randint(0, cnt, (self.n_samples - out.numel(),), generator=g)
                    pad = self.order[lo0 + ridx]
                    out = torch.cat([out, pad], dim=0)

                self.cursor[cell] = 0  
                return out

            batches = 0
            while batches < self.length:
                if subj_ptr + self.n_subjects > self.S:
                    subj_perm = torch.randperm(self.S, generator=g)
                    subj_ptr = 0

                selected = subj_perm[subj_ptr:subj_ptr + self.n_subjects]
                subj_ptr += self.n_subjects

                chunks = []
                for c in range(self.n_classes):
                    base = c * self.S
                    for s in selected.tolist():
                        chunks.append(take_from_cell(base + int(s)))

                batch = torch.cat(chunks, dim=0)
                batch = batch[torch.randperm(batch.numel(), generator=g)]
                yield batch.tolist()
                batches += 1


def create_triplet_loader(x, y, subjects, 
                        batch=BATCH_SIZE, n_classes=CLASSES, 
                        n_subjects=20, persistent_workers=PRESIST_WORKER,
                        pin_memory=PIN_MEMORY, workers=WORKERS):
    sampler = TripletBatchSampler(y, subjects, batch, n_classes, n_subjects)
    return DataLoader(
        TensorDataset(torch.from_numpy(x), 
                      torch.from_numpy(y),
                      torch.from_numpy(subjects)),
        batch_sampler=sampler,
        num_workers=workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory)

def make_triplet_loaders(train_windows, train_meta,
                          val_windows, val_meta, s=N_SUBJECTS, sv=26):
    tl = create_triplet_loader(train_windows, train_meta['classes'], train_meta['subjects'],
                               batch=BATCH_SIZE, n_classes=CLASSES, n_subjects=s)
    vl = create_triplet_loader(val_windows, val_meta['classes'], val_meta['subjects'],
                               batch=BATCH_SIZE, n_classes=CLASSES, n_subjects=sv)
    return tl, vl

# ======== TRAINING & VALIDATING ========
def train(model, train_loader, val_loader, name,
          loss_fn=nn.CrossEntropyLoss(),
          return_emb=False, return_logits=False,
          epochs=EPOCHS, lr=LR_INIT, min_lr=LR_MIN,
          lr_factor=LR_FACTOR, lr_patience=LR_PATIENCE, 
          patience=PATIENCE, device=DEVICE,
          verbose=VERBOSE, save_chkp=False):

    model.to(device)
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr)
    scaler = GradScaler(enabled=(device=="cuda"))

    best_val = 1e9
    best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    wait = 0
    best_epoch = 0

    if save_chkp:
        os.makedirs(f"{CHECKPOINT_PATH}/{name}/", exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0.0, device=device)
        total = 0
        step = 0
        pbar = tqdm(total=len(train_loader), desc=f"{name} | Ep {ep}", 
                    leave=True, dynamic_ncols=True, disable=not verbose)

        for xb, yb, *_ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device=="cuda")):
                if return_emb and return_logits:
                    emb, logits = model(xb, return_emb, return_logits)
                    loss = loss_fn(emb, logits, yb)
                elif return_emb:
                    emb, logits = model(xb, return_emb, return_logits)
                    loss = loss_fn(emb, yb)
                else:
                    logits = model(xb)
                    loss = loss_fn(logits, yb)                    

            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.detach()
            correct += (logits.argmax(1) == yb).sum()
            total += yb.numel()
            step += 1

            if not(step % UPDATE_EVERY):
                pbar.update(UPDATE_EVERY)
                pbar.set_postfix(
                    loss=f"{total_loss.item() / step:10.8f}",
                    acc=f"{correct.item() / max(1, total):6.4f}",
                    LR=f"{opt.param_groups[0]['lr']:8.6f}")

        if step % UPDATE_EVERY:
            pbar.update(step % UPDATE_EVERY)

        val_acc, val_loss, val_bal, val_conf = evaluate(model, val_loader, loss_fn, 
                                            return_emb, return_logits, device)
        sch.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            wait = 0
            best_epoch = ep
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    tqdm.write(f"{name} | Early stop")
                pbar.close()
                break

        pbar.set_postfix(
            loss=f"{total_loss.item() / max(1, len(train_loader)):10.6f}",
            acc=f"{correct.item() / max(1, total):6.4f}",
            val_loss=f"{val_loss:10.6f}",
            val_acc=f"{val_acc:6.4f}",
            val_bal = f"{val_bal:6.4f}", 
            LR=f"{opt.param_groups[0]['lr']:8.6f}",
            wait=f"{wait:3.0f}")
        pbar.close()

        if verbose:
            val_conf_norm = val_conf / val_conf.sum(dim=1, keepdim=True).clamp(min=1.0)
            for i, row in enumerate(val_conf_norm):
                print(f"  c{i}: [" + ", ".join([f"{v.item():.2f}" for v in row]) +
                        f"]  recall={row[i].item():.2f}")
            
        if save_chkp:
            checkpoint = {'epoch': ep,
                        'model_state_dict': model.state_dict()}
            torch.save(checkpoint, f"{CHECKPOINT_PATH}/{name}/chkp_{ep:03d}.pt")

    model.load_state_dict(best_state)
    checkpoint = {'epoch': best_epoch,
                'model_state_dict': model.state_dict()}
    if save_chkp:
        torch.save(checkpoint, f"{CHECKPOINT_PATH}/{name}/{name}.pt")

    return model


def train_grl(model, train_loader, val_loader, name,
            loss_fn=nn.CrossEntropyLoss(),
            loss_fn_sbj=None,
            return_emb=False, return_logits=False,
            grl_weight=1.0, ramp_epochs=50,
            epochs=EPOCHS, lr=LR_INIT, min_lr=LR_MIN,
            lr_factor=LR_FACTOR, lr_patience=LR_PATIENCE, 
            patience=PATIENCE, device=DEVICE,
            verbose=VERBOSE, save_chkp=False):

    model.to(device)
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr)
    scaler = GradScaler(enabled=(device=="cuda"))

    max_steps = max(1, ramp_epochs * len(train_loader))
    global_steps = 0
    best_val = 1e9
    best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    wait = 0
    best_epoch = 0

    if save_chkp:
        os.makedirs(f"{CHECKPOINT_PATH}/{name}/", exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        c_loss = torch.tensor(0.0, device=device)
        grl_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0.0, device=device)
        total = 0
        step = 0
        pbar = tqdm(total=len(train_loader), desc=f"{name} | Ep {ep}", 
                    leave=True, dynamic_ncols=True, disable=not verbose)

        for xb, yb, ys in train_loader:
            p = global_steps / max_steps
            lmbd = np.clip(p, 0, 1)
            model.grl.lambd = float(lmbd)

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device=="cuda")):
                logits, logits_grl = model(xb, return_grl=True)
                loss_c = loss_fn(logits, yb)
                loss_grl = loss_fn_sbj(logits_grl, ys)
                loss = loss_c + grl_weight * loss_grl

            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.detach()
            c_loss += loss_c.detach()
            grl_loss += loss_grl.detach()
            correct += (logits.argmax(1) == yb).sum()
            total += yb.numel()
            step += 1
            global_steps += 1

            if not(step % UPDATE_EVERY):
                pbar.update(UPDATE_EVERY)
                pbar.set_postfix(
                    loss=f"{total_loss.item() / step:10.8f}",
                    c_loss=f"{c_loss.item() / step:10.8f}",
                    grl_loss=f"{grl_loss.item() / step:10.8f}",
                    acc=f"{correct.item() / max(1, total):6.4f}",
                    LR=f"{opt.param_groups[0]['lr']:8.6f}")

        if step % UPDATE_EVERY:
            pbar.update(step % UPDATE_EVERY)

        val_acc, val_loss, val_bal, val_conf = evaluate(model, val_loader, loss_fn, 
                                                return_emb, return_logits, device)
        sch.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            wait = 0
            best_epoch = ep
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    tqdm.write(f"{name} | Early stop")
                pbar.close()
                break

        pbar.set_postfix(
            loss=f"{total_loss.item() / max(1, len(train_loader)):10.6f}",
            c_loss=f"{c_loss.item() / max(1, len(train_loader)):10.8f}",
            grl_loss=f"{grl_loss.item() / max(1, len(train_loader)):10.8f}",
            acc=f"{correct.item() / max(1, total):6.4f}",
            val_loss=f"{val_loss:10.6f}",
            val_acc=f"{val_acc:6.4f}",
            val_bal = f"{val_bal:6.4f}", 
            LR=f"{opt.param_groups[0]['lr']:8.6f}",
            wait=f"{wait:3.0f}")
        pbar.close()

        if verbose:
            val_conf_norm = val_conf / val_conf.sum(dim=1, keepdim=True).clamp(min=1.0)
            for i, row in enumerate(val_conf_norm):
                print(f"  c{i}: [" + ", ".join([f"{v.item():.2f}" for v in row]) +
                        f"]  recall={row[i].item():.2f}")

        if save_chkp:
            checkpoint = {'epoch': ep,
                        'model_state_dict': model.state_dict()}
            torch.save(checkpoint, f"{CHECKPOINT_PATH}/{name}/chkp_{ep:03d}.pt")

    model.load_state_dict(best_state)
    checkpoint = {'epoch': best_epoch,
                'model_state_dict': model.state_dict()}
    torch.save(checkpoint, f"{CHECKPOINT_PATH}/{name}/{name}.pt")
    return model


def train_sbj(model, train_loader, val_loader, name,
          loss_fn=nn.CrossEntropyLoss(),
          return_emb=False, return_logits=False,
          epochs=EPOCHS, lr=LR_INIT, min_lr=LR_MIN,
          lr_factor=LR_FACTOR, lr_patience=LR_PATIENCE, 
          patience=PATIENCE, device=DEVICE,
          verbose=VERBOSE, save_chkp=False):

    model.to(device)
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr)
    scaler = GradScaler(enabled=(device=="cuda"))

    best_val = 1e9
    best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    wait = 0
    best_epoch = 0

    if save_chkp:
        os.makedirs(f"{CHECKPOINT_PATH}/{name}/", exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0.0, device=device)
        total = 0
        step = 0
        pbar = tqdm(total=len(train_loader), desc=f"{name} | Ep {ep}", 
                    leave=True, dynamic_ncols=True, disable=not verbose)

        for xb, yb, ys in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device=="cuda")):
                if return_emb and return_logits:
                    emb, logits = model(xb, return_emb, return_logits)
                    loss = loss_fn(emb, logits, yb, ys)
                if return_emb:
                    emb, logits = model(return_emb, return_logits)
                    loss = loss_fn(emb, logits, yb)
                else:
                    logits = model(xb)
                    loss = loss_fn(logits, yb, ys)                    

            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.detach()
            correct += (logits.argmax(1) == yb).sum()
            total += yb.numel()
            step += 1

            if not(step % UPDATE_EVERY):
                pbar.update(UPDATE_EVERY)
                pbar.set_postfix(
                    loss=f"{total_loss.item() / step:10.8f}",
                    acc=f"{correct.item() / max(1, total):6.4f}",
                    LR=f"{opt.param_groups[0]['lr']:8.6f}")

        if step % UPDATE_EVERY:
            pbar.update(step % UPDATE_EVERY)

        val_acc, val_loss, val_bal, val_conf = evaluate_sbj(model, val_loader, loss_fn, 
                                     return_emb, return_logits, device)
        sch.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            wait = 0
            best_epoch = ep
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    tqdm.write(f"{name} | Early stop")
                pbar.close()
                break

        pbar.set_postfix(
            loss=f"{total_loss.item() / max(1, len(train_loader)):10.6f}",
            acc=f"{correct.item() / max(1, total):6.4f}",
            val_loss=f"{val_loss:10.6f}",
            val_acc=f"{val_acc:6.4f}",
            val_bal = f"{val_bal:6.4f}", 
            LR=f"{opt.param_groups[0]['lr']:8.6f}",
            wait=f"{wait:3.0f}")
        pbar.close()

        if verbose:
            val_conf_norm = val_conf / val_conf.sum(dim=1, keepdim=True).clamp(min=1.0)
            for i, row in enumerate(val_conf_norm):
                print(f"  c{i}: [" + ", ".join([f"{v.item():.2f}" for v in row]) +
                        f"]  recall={row[i].item():.2f}")

        if save_chkp:
            checkpoint = {'epoch': ep,
                        'model_state_dict': model.state_dict()}
            torch.save(checkpoint, f"{CHECKPOINT_PATH}/{name}/chkp_{ep:03d}.pt")

    model.load_state_dict(best_state)
    checkpoint = {'epoch': best_epoch,
                'model_state_dict': model.state_dict()}
    if save_chkp:
        torch.save(checkpoint, f"{CHECKPOINT_PATH}/{name}/{name}.pt")
    return model


def train_triplet(model, train_loader, val_loader, name,
                  criterion_ce=nn.CrossEntropyLoss(), criterion_tri=None,
                  epochs=EPOCHS, lr=LR_INIT, min_lr=LR_MIN,
                  lr_factor=LR_FACTOR, lr_patience=LR_PATIENCE, 
                  patience=PATIENCE, device=DEVICE, verbose=VERBOSE,
                  alpha_start=0.0, alpha_end=0.2, warmup_epochs=20,
                  save_chkp=False):

    model.to(device)
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr)
    scaler = GradScaler(enabled=(device=="cuda"))
    
    best_val_metric = 1e9
    best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    wait = 0
    best_epoch = 0

    if save_chkp:
        os.makedirs(f"{CHECKPOINT_PATH}/{name}/", exist_ok=True)

    ep = 1
    while ep <= epochs:

        progress = min((ep - 1) / warmup_epochs, 1.0)
        sig = 1 / (1 + math.exp(-10 * (progress - 0.65))) 
        sig_0 = 1 / (1 + math.exp(-10 * (0 - 0.65)))
        sig_1 = 1 / (1 + math.exp(-10 * (1 - 0.65)))
        current_alpha = alpha_start + (alpha_end - alpha_start) * (sig - sig_0) / (sig_1 - sig_0)
        current_ce_w = 1.0 - current_alpha

        model.train()
        total_loss = torch.tensor(0.0, device=device)
        total_ce = torch.tensor(0.0, device=device)
        total_tri = torch.tensor(0.0, device=device)
        correct = torch.tensor(0.0, device=device)
        total = 0
        step = 0
        
        desc = f"{name} | Ep {ep} [alpha={current_alpha:.3f}]"
        pbar = tqdm(total=len(train_loader), desc=desc, 
                    leave=True, dynamic_ncols=True, disable=not verbose)

        for xb, yb, ys in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device=="cuda")):
                emb, logits = model(xb, return_emb=True, return_logits=True)
                
                loss_ce = criterion_ce(logits, yb)
                loss_tri = criterion_tri(emb, yb, ys)
                
                loss = (current_ce_w * loss_ce) + (current_alpha * loss_tri)

            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.detach()
            total_ce += loss_ce.detach()
            total_tri += loss_tri.detach()
            
            correct += (logits.argmax(1) == yb).sum()
            total += yb.numel()
            step += 1

            if not(step % UPDATE_EVERY):
                pbar.update(UPDATE_EVERY)
                pbar.set_postfix(
                    L=f"{total_loss.item()/step:.6f}",
                    CE=f"{total_ce.item()/step:.6f}",
                    TR=f"{total_tri.item()/step:.6f}",
                    acc=f"{correct.item()/max(1, total):.3f}")

        if step % UPDATE_EVERY:
            pbar.update(step % UPDATE_EVERY)

        val_acc, val_loss_ce, val_loss_tri, val_bal, val_conf = evaluate_triplet(
            model, val_loader, criterion_ce, device, 
            triplet_fn=criterion_tri, alpha=current_alpha)
        
        monitor_metric = (current_ce_w * val_loss_ce) + (current_alpha * val_loss_tri)

        if ep > warmup_epochs:
            sch.step(monitor_metric)
            if monitor_metric < best_val_metric:
                best_val_metric = monitor_metric
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                wait = 0
                best_epoch = ep
            else:
                wait += 1
                if wait >= patience:
                    tqdm.write(f"{name} | Early stop")
                    pbar.close()
                    break

        pbar.set_postfix(
            L=f"{total_loss.item()/step:.6f}",
            CE=f"{total_ce.item()/step:.6f}",
            TR=f"{total_tri.item()/step:.6f}",
            acc=f"{correct.item()/max(1, total):.3f}",
            v_CE=f"{val_loss_ce:.6f}",
            v_TR=f"{val_loss_tri:.6f}",
            v_Acc=f"{val_acc:.3f}",
            v_bal = f"{val_bal:6.4f}", 
            v_L = f"{monitor_metric:6.4f}", 
            LR=f"{opt.param_groups[0]['lr']:.6f}",
            wait=f"{wait:2d}")
        pbar.close()

        if verbose:
            val_conf_norm = val_conf / val_conf.sum(dim=1, keepdim=True).clamp(min=1.0)
            for i, row in enumerate(val_conf_norm):
                print(f"  c{i}: [" + ", ".join([f"{v.item():.2f}" for v in row]) +
                        f"]  recall={row[i].item():.2f}")
                
        if save_chkp:
            checkpoint = {'epoch': ep,
                          'model_state_dict': model.state_dict()}
            torch.save(checkpoint, f"{CHECKPOINT_PATH}/{name}/chkp_{ep:03d}.pt")

        ep += 1

    model.load_state_dict(best_state)
    checkpoint = {'epoch': best_epoch,
                'model_state_dict': model.state_dict()}
    torch.save(checkpoint, f"{CHECKPOINT_PATH}/{name}/{name}.pt")
    return model


# ---- EMBEDDING PCA CALLBACK ----
# PCA 
class PCA_GPU:
    def __init__(self, dims=2, device=DEVICE):
        self.device = device
        self.dims   = dims
        self.mean_  = None
        self.components_ = None

    def fit(self, X: torch.Tensor):
        # Caller guarantees X is already on self.device
        N = X.shape[0]
        self.mean_ = X.mean(dim=0, keepdim=True)
        Xc = X - self.mean_                          # N × D
        C  = torch.mm(Xc.T, Xc).div_(N - 1)         # D × D, in-place div
        _, eigvecs = torch.linalg.eigh(C)            # ascending eigenvalues
        # eigh is already sorted ascending -> last `dims` cols = top components
        self.components_ = eigvecs[:, -self.dims:].flip(1).contiguous()
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean_) @ self.components_   # N x dims

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)


# Collect embeddings
@torch.no_grad()
def collect_embeddings(model, loader, device):
    model.eval()
    N = len(loader.dataset)
    is_cuda = (device == "cuda")

    # Infer D from a single sample 
    sample_xb, *_ = next(iter(loader))
    with autocast(device_type="cuda", enabled=is_cuda):
        sample_emb = model(sample_xb[:1].to(device), return_emb=True)
    D = sample_emb.shape[1]

    # Pinned CPU buffers - enables async DMA from GPU
    feats  = torch.empty(N, D, dtype=torch.float32,
                         pin_memory=is_cuda)
    labels = torch.empty(N,    dtype=torch.long,
                         pin_memory=is_cuda)

    ptr = 0
    for xb, yb, *_ in loader:
        b   = xb.size(0)
        xb  = xb.to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=is_cuda):
            emb = model(xb, return_emb=True)          # GPU

        # non_blocking=True: DMA into pinned memory, GPU keeps going
        feats [ptr:ptr+b].copy_(emb, non_blocking=True)
        labels[ptr:ptr+b].copy_(yb)                   # already CPU
        ptr += b

    if is_cuda:
        torch.cuda.synchronize()                      # wait for all DMA

    return (feats .to(device, non_blocking=True),
            labels.to(device, non_blocking=True))


_GESTURE_NAMES = ['Rest', 'Hand Close', 'Flexion', 'Extension', 'Hand Open']
_PALETTE       = ['#888888', '#4C72B0', '#DD8452', '#55A868', '#C44E52']
_Z_ORDER       = [1, 2, 3, 4, 0]   # rest drawn last, sits on top at the fan origin
def _plot_epoch(Z, y, title, path):
    dims  = Z.shape[1]
    n_cls = len(_PALETTE)

    fig, axes = plt.subplots(dims, dims, figsize=(2 * dims, 2 * dims), dpi=150)
    fig.subplots_adjust(hspace=0.06, wspace=0.06,
                        left=0.10, right=0.97, top=0.90, bottom=0.12)
    axes = np.atleast_2d(axes)

    scatter_kw = dict(s=5, alpha=0.55, linewidths=0, rasterized=True)

    for i in range(dims):
        for j in range(dims):
            ax = axes[i, j]
            if i == j:
                # diagonal - per-PC marginal histograms
                for cls in range(n_cls):
                    mask = y == cls
                    if mask.any():
                        ax.hist(Z[mask, i], bins=28, color=_PALETTE[cls],
                                alpha=0.55, linewidth=0, density=True)
            elif i > j:
                # off-diagonal - scatter of the single embedding projection
                for cls in _Z_ORDER:
                    mask = y == cls
                    if mask.any():
                        ax.scatter(Z[mask, j], Z[mask, i],
                                   color=_PALETTE[cls], **scatter_kw)

            else: ax.set_visible(False)

            # spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.4)
            ax.spines['bottom'].set_linewidth(0.4)
            # ticks
            ax.tick_params(labelsize=6, length=2, width=0.4, pad=2)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3, prune='both'))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune='both'))
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.tick_params(bottom=False, left=False)
            if i == dims - 1:
                ax.set_xlabel(f"PC{j + 1}", fontsize=16)
            if j == 0:
                ax.set_ylabel(f"PC{i + 1}", fontsize=16)

    # legend
    handles = [
        mpatches.Patch(color=_PALETTE[c], label=_GESTURE_NAMES[c])
        for c in range(n_cls)
    ]
    fig.legend(handles=handles, fontsize=12, frameon=False,
            loc='lower center', ncol=n_cls,
            bbox_to_anchor=(0.53, 0.00),
            handlelength=1.0, handleheight=0.85,
            columnspacing=1.2)

    # fig.suptitle(title, fontsize=16, y=0.96)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

# PCA sweep
@torch.no_grad()
def run_pca_sweep(model, loader, name, dims=2,
                  device=DEVICE, n_plot_workers=4):
    checkpoint_dir = f"{CHECKPOINT_PATH}/{name}/"
    output_dir     = f"{FIGURE_PATH}/{name}_PCAs_{dims}"
    os.makedirs(output_dir, exist_ok=True)

    epoch_files = sorted(
        f for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt") and name not in f   # skip the "best" / named ckpt
    )
    if not epoch_files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    model.to(device).eval()

    # Pass 1: fit PCA on last epoch
    best = torch.load(f"{checkpoint_dir}/{name}.pt", map_location=device)
    model.load_state_dict(best["model_state_dict"])
    best_ep = best["epoch"]

    feats, _ = collect_embeddings(model, loader, device)
    pca = PCA_GPU(dims=dims, device=device).fit(feats)
    del feats; torch.cuda.empty_cache()

    # Pass 2: stream + transform + plot (plot in background thread)
    with ThreadPoolExecutor(max_workers=n_plot_workers) as pool:
        for i, f in enumerate(epoch_files):
            print(f"{i+1}/{len(epoch_files)}")
            ckpt  = torch.load(f"{checkpoint_dir}/{f}", map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            epoch = ckpt["epoch"]

            if epoch != best_ep:
                continue

            feats, labels = collect_embeddings(model, loader, device)
            Z = pca.transform(feats).cpu().numpy()   # GPU matmul -> CPU once
            y = labels.cpu().numpy()
            del feats, labels; torch.cuda.empty_cache()

            # Fire-and-forget: savefig runs while GPU is already on next epoch
            pool.submit(
                _plot_epoch, Z, y,
                f"{name} | Epoch {epoch}",
                f"{output_dir}/emb_{name}_ep_{epoch:03d}.png")
        # ThreadPoolExecutor.__exit__ joins all pending saves before returning
            if epoch == best_ep:
                break

    print(f"PCA sweep done: {output_dir}")


# ---- VALIDATION ----
@torch.no_grad()
def evaluate(model, loader, loss_fn, 
             return_emb, return_logits, device):
    model.eval()
    # Initialize on GPU
    lsum = torch.tensor(0.0, device=device)
    cor = torch.tensor(0.0, device=device)
    tot = torch.tensor(0,   device=device, dtype=torch.long)
    class_cor   = torch.zeros(CLASSES, device=device)   # per-class correct
    class_tot   = torch.zeros(CLASSES, device=device)   # per-class total
    val_conf_matrix = torch.zeros((CLASSES, CLASSES), device=device)

    for xb, yb, *_ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
            if return_emb and return_logits:
                emb, logits = model(xb, return_emb, return_logits)
                loss = loss_fn(emb, logits, yb)
            else:
                logits = model(xb)
                loss = loss_fn(logits, yb)  

        preds = logits.argmax(1)
        lsum += loss.detach()
        cor += (preds == yb).sum()
        tot += yb.numel()
        hits = (preds == yb).float()
        class_cor.scatter_add_(0, yb, hits)
        class_tot.scatter_add_(0, yb, torch.ones_like(hits))
        idx = (yb * CLASSES + preds).clamp(0, CLASSES * CLASSES - 1)
        batch_counts = torch.bincount(idx, minlength=CLASSES * CLASSES).float().view(CLASSES, CLASSES)
        val_conf_matrix += batch_counts
        mask = class_tot > 0
        bal_acc = (class_cor[mask] / class_tot[mask]).mean().item()

    return (cor.item() / max(1, tot), 
            lsum.item() / max(1, len(loader)),
            bal_acc, val_conf_matrix)


@torch.no_grad()
def evaluate_sbj(model, loader, loss_fn, 
             return_emb, return_logits, device):
    model.eval()
    # Initialize on GPU
    lsum = torch.tensor(0.0, device=device)
    cor = torch.tensor(0.0, device=device)
    tot = torch.tensor(0,   device=device, dtype=torch.long)
    class_cor = torch.zeros(CLASSES, device=device)   # per-class correct
    class_tot = torch.zeros(CLASSES, device=device)   # per-class total
    val_conf_matrix = torch.zeros((CLASSES, CLASSES), device=device)

    for xb, yb, ys in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
            if return_emb and return_logits:
                emb, logits = model(xb, return_emb, return_logits)
                loss = loss_fn(emb, logits, yb, ys)
            else:
                logits = model(xb)
                loss = loss_fn(logits, yb, ys)  

        preds = logits.argmax(1)
        lsum += loss.detach()
        cor += (preds == yb).sum()
        tot += yb.numel()
        hits = (preds == yb).float()
        class_cor.scatter_add_(0, yb, hits)
        class_tot.scatter_add_(0, yb, torch.ones_like(hits))
        idx = (yb * CLASSES + preds).clamp(0, CLASSES * CLASSES - 1)
        batch_counts = torch.bincount(idx, minlength=CLASSES * CLASSES).float().view(CLASSES, CLASSES)
        val_conf_matrix += batch_counts
        mask = class_tot > 0
        bal_acc = (class_cor[mask] / class_tot[mask]).mean().item()

    return (cor.item() / max(1, tot), 
            lsum.item() / max(1, len(loader)),
            bal_acc, val_conf_matrix)


@torch.no_grad()
def evaluate_triplet(model, loader, loss_fn, 
            device, triplet_fn=None, alpha=0.0):
    model.eval()
    lsum = torch.tensor(0.0, device=device)
    cor = torch.tensor(0.0, device=device)
    tri_sum = torch.tensor(0.0, device=device)
    tot = torch.tensor(0,   device=device, dtype=torch.long)
    class_cor = torch.zeros(CLASSES, device=device)   # per-class correct
    class_tot = torch.zeros(CLASSES, device=device)   # per-class total
    val_conf_matrix = torch.zeros((CLASSES, CLASSES), device=device)

    for xb, yb, sb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        sb = sb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
            emb, logits = model(xb, return_emb=True, return_logits=True)
            
            loss = loss_fn(logits, yb)
            
            if triplet_fn is not None and alpha > 0:
                t_loss = triplet_fn(emb, yb, sb)
                tri_sum += t_loss.detach()
                
        lsum += loss.detach()
        cor += (logits.argmax(1) == yb).sum()
        tot += yb.numel()
        
    avg_ce = lsum.item() / max(1, len(loader))
    avg_tri = tri_sum.item() / max(1, len(loader))
    acc = cor.item() / max(1, tot)

    preds = logits.argmax(1)
    hits = (preds == yb).float()
    class_cor.scatter_add_(0, yb, hits)
    class_tot.scatter_add_(0, yb, torch.ones_like(hits))
    idx = (yb * CLASSES + preds).clamp(0, CLASSES * CLASSES - 1)
    batch_counts = torch.bincount(idx, minlength=CLASSES * CLASSES).float().view(CLASSES, CLASSES)
    val_conf_matrix += batch_counts
    mask = class_tot > 0
    bal_acc = (class_cor[mask] / class_tot[mask]).mean().item()
    
    return (acc, avg_ce, avg_tri,
            bal_acc, val_conf_matrix)


# ======== GENERAL TESTING ========
@torch.no_grad()
def eval_test(model, loaders, metas, name,
              save=True, multi_head=None,
              csv_path = RESULTS_PATH,
              device=DEVICE):

    model.to(device)
    model.eval()
    results = {}

    if save:
        os.makedirs(f"{FIGURE_PATH}/{name}/", exist_ok=True)

    def run(loader, meta, tag):
        N = len(loader.dataset)
        # Pre-allocate on GPU to avoid dynamic growth
        preds = torch.empty(N, dtype=torch.long, device=device)
        ptr = 0
        for xb, *_ in loader:
            b = xb.size(0)
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
                out = model(xb)
                if multi_head is not None:
                    out = out[multi_head]
            preds[ptr:ptr+b] = out.argmax(1)
            ptr += b
        
        # Single sync point
        preds = preds.cpu().numpy()
        subjects = np.asarray(meta['subjects'])
        labels = np.asarray(meta['classes'])
        unique_subjects = np.unique(subjects)
        n_subj = len(unique_subjects)
        
        acc, act_acc, bal_acc, f1 = np.zeros(n_subj), np.zeros(n_subj), \
                                np.zeros(n_subj), np.zeros(n_subj)
        
        for i, s in enumerate(unique_subjects):
            mask = (subjects == s)
            ps, ls = preds[mask], labels[mask]

            # CA (Classification Accuracy)
            acc[i] = (ps == ls).mean()

            f1[i] = f1_score(ls, ps, average='macro')

            # AER logic (Active Error Rate / Active Accuracy)
            act_mask = (ls != 0)
            if act_mask.any():
                act_acc[i] = (ps[act_mask] == ls[act_mask]).mean()

            # Vectorized Balanced Accuracy
            # Efficiently calculates recall for all classes at once
            cm = confusion_matrix(ls, ps, labels=np.arange(CLASSES))
            with np.errstate(divide='ignore', invalid='ignore'):
                per_class = np.diag(cm) / cm.sum(axis=1)
                bal_acc[i] = np.nanmean(per_class)

        acc, act_acc, bal_acc = acc * 100, act_acc * 100, bal_acc * 100

        if save:
            fig, axs = plt.subplots(2, 2, figsize=(11, 11), dpi=200)
            ax1, ax2, ax3, ax4 = axs.flatten()
            fig.suptitle(
                f"{tag} | Mean Acc {acc.mean():.2f} ± {np.std(acc):.2f} "
                f"| Mean Actv {act_acc.mean():.2f} ± {np.std(act_acc):.2f} "
                f"| Mean Bal {bal_acc.mean():.2f} ± {np.std(bal_acc):.2f} "
                f"| Mean F1 {f1.mean():.2f} ± {np.std(f1):.2f}"
            )
            
            _idx = np.argsort(bal_acc)

            ax1.bar(np.arange(n_subj), acc[_idx])
            ax1.axhline(acc.mean(), color='red', linestyle='--')
            ax1.set_title('Per Subject Accuracy')
            
            ax2.bar(np.arange(n_subj), act_acc[_idx])
            ax2.axhline(act_acc.mean(), color='red', linestyle='--')
            ax2.set_title('Per Subject Active Accuracy')

            ax3.bar(np.arange(n_subj), bal_acc[_idx])
            ax3.axhline(bal_acc.mean(), color='red', linestyle='--')
            ax3.set_title('Per Subject Balanced Accuracy')

            ax4.bar(np.arange(n_subj), f1[_idx])
            ax4.axhline(f1.mean(), color='red', linestyle='--')
            ax4.set_title('Per F1 Score')

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            
            fig.savefig(f"{FIGURE_PATH}/{name}/{tag}.jpg")
            fig.clf()
            plt.close(fig)

            os.makedirs(f"{CHECKPOINT_PATH}/{name}/", exist_ok=True)
            np.save(f"{CHECKPOINT_PATH}/{name}/results_{tag}.npy", 
                    np.stack((acc, act_acc, bal_acc, f1)))
            np.save(f"{CHECKPOINT_PATH}/{name}/preds_{tag}.npy", preds)
            np.save(f"{CHECKPOINT_PATH}/{name}/labels_{tag}.npy", labels)

        return {"acc_mean": acc.mean(), "acc_std": acc.std(),
                "act_acc_mean": act_acc.mean(), "act_acc_std": act_acc.std(),
                "bal_acc_mean": bal_acc.mean(), "bal_acc_std": bal_acc.std(),
                "F1": f1.mean(), "F1_std": f1.std()}

    # Iterate through provided loaders (raw, segmented, relabeled)
    for tag in loaders.keys():
        results[tag] = run(loaders[tag], metas[tag], tag)

    # Atomic CSV logging (Fastest concurrent-safe method)
    rows = [{"model": name, "test_set": tag, **r} for tag, r in results.items()]
    # empty_row = {k: "" for k in rows[0].keys()}
    df_new = pd.DataFrame(rows)
    df_new.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))

    return results


@torch.no_grad()
def eval_within(model, loader, meta,
                multi_head=None,
                device=DEVICE):

    model.to(device)
    model.eval()
    results = {}

    def run(loader, meta):
        N = len(loader.dataset)
        # Pre-allocate on GPU to avoid dynamic growth
        preds = torch.empty(N, dtype=torch.long, device=device)
        ptr = 0
        for xb, *_ in loader:
            b = xb.size(0)
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
                out = model(xb)
                if multi_head is not None:
                    out = out[multi_head]
            preds[ptr:ptr+b] = out.argmax(1)
            ptr += b
        
        # Single sync point
        preds = preds.cpu().numpy()
        labels = np.asarray(meta['classes'])
        
        ps, ls = preds, labels

        f1 = f1_score(ls, ps, average='macro')

        # CA (Classification Accuracy)
        acc = (ps == ls).mean()

        # AER logic (Active Error Rate / Active Accuracy)
        act_mask = (ls != 0)
        if act_mask.any():
            act_acc = (ps[act_mask] == ls[act_mask]).mean()

        # Vectorized Balanced Accuracy
        # Efficiently calculates recall for all classes at once
        cm = confusion_matrix(ls, ps, labels=np.arange(CLASSES))
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class = np.diag(cm) / cm.sum(axis=1)
            bal_acc = np.nanmean(per_class)

        acc, act_acc, bal_acc = acc * 100, act_acc * 100, bal_acc * 100

        return {"acc_mean": acc.mean(),
                "act_acc_mean": act_acc.mean(),
                "bal_acc_mean": bal_acc.mean(),
                "F1": f1}

    # Iterate through provided loaders (raw, segmented, relabeled)
    results = run(loader, meta)

    return results


@torch.no_grad()
def eval_within_lda(model, x, meta):

    results = {}

    def run(x, meta):
        preds = model.predict(x)
        labels = np.asarray(meta['classes'])
        ps, ls = preds, labels

        # CA (Classification Accuracy)
        acc = (ps == ls).mean()

        # AER logic (Active Error Rate / Active Accuracy)
        act_mask = (ls != 0)
        if act_mask.any():
            act_acc = (ps[act_mask] == ls[act_mask]).mean()

        # Vectorized Balanced Accuracy
        # Efficiently calculates recall for all classes at once
        cm = confusion_matrix(ls, ps, labels=np.arange(CLASSES))
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class = np.diag(cm) / cm.sum(axis=1)
            bal_acc = np.nanmean(per_class)

        acc, act_acc, bal_acc = acc * 100, act_acc * 100, bal_acc * 100

        return {"acc_mean": acc.mean(),
                "act_acc_mean": act_acc.mean(),
                "bal_acc_mean": bal_acc.mean()}

    # Iterate through provided loaders (raw, segmented, relabeled)
    results = run(x, meta)

    return results


def eval_test_lda(model, X, metas, name, save=True,
                  csv_path=RESULTS_PATH):
    results = {}
    os.makedirs(f"{FIGURE_PATH}/{name}/", exist_ok=True)

    def run(_x, meta, tag):
        preds = model.predict(_x)

        subjects = np.asarray(meta['subjects'])
        labels   = np.asarray(meta['classes'])
        unique_subjects = np.unique(subjects)
        n_subj = len(unique_subjects)

        acc, act_acc, bal_acc, f1 = (np.zeros(n_subj) for _ in range(4))

        for i, s in enumerate(unique_subjects):
            mask = (subjects == s)
            ps, ls = preds[mask], labels[mask]

            acc[i] = (ps == ls).mean()

            f1[i] = f1_score(ls, ps, average='macro')

            act_mask = (ls != 0)
            if act_mask.any():
                act_acc[i] = (ps[act_mask] == ls[act_mask]).mean()

            cm = confusion_matrix(ls, ps, labels=np.arange(CLASSES))
            with np.errstate(divide='ignore', invalid='ignore'):
                per_class = np.diag(cm) / cm.sum(axis=1)
                bal_acc[i] = np.nanmean(per_class)

        acc, act_acc, bal_acc = acc * 100, act_acc * 100, bal_acc * 100

        fig, axs = plt.subplots(2, 2, figsize=(11, 11), dpi=200)
        ax1, ax2, ax3, ax4 = axs.flatten()
        fig.suptitle(
            f"{tag} | Mean Acc {acc.mean():.2f} ± {acc.std():.2f} "
            f"| Mean Actv {act_acc.mean():.2f} ± {act_acc.std():.2f} "
            f"| Mean Bal {bal_acc.mean():.2f} ± {bal_acc.std():.2f} "
            f"| Mean F1 {f1.mean():.2f} ± {f1.std():.2f}"
        )

        _idx = np.argsort(bal_acc)

        ax1.bar(np.arange(n_subj), acc[_idx])
        ax1.axhline(acc.mean(), color='red', linestyle='--')
        ax1.set_title('Per Subject Accuracy')

        ax2.bar(np.arange(n_subj), act_acc[_idx])
        ax2.axhline(act_acc.mean(), color='red', linestyle='--')
        ax2.set_title('Per Subject Active Accuracy')

        ax3.bar(np.arange(n_subj), bal_acc[_idx])
        ax3.axhline(bal_acc.mean(), color='red', linestyle='--')
        ax3.set_title('Per Subject Balanced Accuracy')

        ax4.bar(np.arange(n_subj), f1[_idx])
        ax4.axhline(f1.mean(), color='red', linestyle='--')
        ax4.set_title('Per Subject F1 Score')

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f"{FIGURE_PATH}/{name}/{tag}.jpg")
        fig.clf()
        plt.close(fig)

        if save:
            os.makedirs(f"{CHECKPOINT_PATH}/{name}/", exist_ok=True)
            np.save(f"{CHECKPOINT_PATH}/{name}/results_{tag}.npy",
                    np.stack((acc, act_acc, bal_acc, f1)))
            np.save(f"{CHECKPOINT_PATH}/{name}/preds_{tag}.npy", preds)
            np.save(f"{CHECKPOINT_PATH}/{name}/labels_{tag}.npy", labels)

        return {"acc_mean": acc.mean(), "acc_std": acc.std(),
                "act_acc_mean": act_acc.mean(), "act_acc_std": act_acc.std(),
                "bal_acc_mean": bal_acc.mean(), "bal_acc_std": bal_acc.std(),
                "F1": f1.mean(), "F1_std": f1.std()}

    for tag in X.keys():
        results[tag] = run(X[tag], metas[tag], tag)

    rows = [{"model": name, "test_set": tag, **r} for tag, r in results.items()]
    df_new = pd.DataFrame(rows)
    df_new.to_csv(csv_path, mode='a', index=False,
                  header=not os.path.exists(csv_path))

    return results


# ---- DDP-aware loader ----
def create_loader_ddp(x, y, s, batch, rank, world_size,
                      shuffle=True, workers=WORKERS,
                      prefetch_factor=PRE_FETCH,
                      pin_memory=PIN_MEMORY):
    sampler = DistributedSampler(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(s)),
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=True)      # keeps batch sizes uniform across ranks
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(s)),
        batch_size=batch,
        sampler=sampler,     # shuffle=False because sampler handles it
        num_workers=workers,
        prefetch_factor=prefetch_factor if workers > 0 else None,
        pin_memory=pin_memory,
        drop_last=True)


def train_ddp(model, train_loader, val_loader, name,
              loss_fn=nn.CrossEntropyLoss(),
              return_emb=False, return_logits=False,
              epochs=EPOCHS, lr=LR_INIT, min_lr=LR_MIN,
              lr_factor=LR_FACTOR, lr_patience=LR_PATIENCE,
              patience=PATIENCE, device=None,
              verbose=VERBOSE, save_chkp=False,
              rank=0, world_size=1):

    if device is None:
        device = next(model.parameters()).device

    IS_MAIN = (rank == 0)

    # Optimizer over unwrapped params
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr)
    scaler = GradScaler(enabled=True)

    best_val = 1e9
    best_state = {k: v.clone().cpu() for k, v in model.module.state_dict().items()}
    wait = 0
    best_epoch = 0

    if IS_MAIN and save_chkp:
        os.makedirs(f"{CHECKPOINT_PATH}/{name}/", exist_ok=True)

    for ep in range(1, epochs + 1):
        train_loader.sampler.set_epoch(ep)

        model.train()
        loss_fn.train()
        total_loss = torch.tensor(0.0, device=device)
        correct    = torch.tensor(0.0, device=device)
        total = 0; step = 0

        pbar = tqdm(total=len(train_loader), desc=f"{name} | Ep {ep}",
                    leave=True, dynamic_ncols=True,
                    disable=(not verbose or not IS_MAIN))

        for xb, yb, ys in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                if return_emb and return_logits:
                    emb, logits = model(xb, return_emb, return_logits)
                    loss = loss_fn(emb, logits, yb, ys)
                elif return_emb:
                    emb = model(xb, return_emb, return_logits)
                    loss = loss_fn(emb, yb)
                else:
                    logits = model(xb)
                    loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.detach()
            if return_logits:
                correct += (logits.argmax(1) == yb).sum()
            total      += yb.numel()
            step       += 1

            if IS_MAIN and not (step % UPDATE_EVERY):
                pbar.update(UPDATE_EVERY)
                pbar.set_postfix(
                    loss=f"{total_loss.item()/step:10.8f}",
                    acc=f"{correct.item()/max(1,total):6.4f}",
                    LR=f"{opt.param_groups[0]['lr']:8.6f}")

        if IS_MAIN and step % UPDATE_EVERY:
            pbar.update(step % UPDATE_EVERY)

        # ---- Validation: gather metrics from ALL ranks ----
        val_acc, val_loss, val_bal, val_conf, val_std = evaluate_ddp(model, val_loader, loss_fn,
                                                        return_emb, return_logits,
                                                        device, world_size, return_std=True)
        sch.step(val_loss)

        if IS_MAIN:
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone().cpu()
                              for k, v in model.module.state_dict().items()}
                wait = 0
                best_epoch = ep
            else:
                wait += 1

            pbar.set_postfix(
                loss=f"{total_loss.item()/max(1,len(train_loader)):10.6f}",
                acc=f"{correct.item()/max(1,total):6.4f}",
                val_loss=f"{val_loss:10.6f}",
                val_acc=f"{val_acc:6.4f}",
                val_bal = f"{val_bal:6.4f}", 
                val_std = f"{val_std:6.4f}", 
                LR=f"{opt.param_groups[0]['lr']:8.6f}",
                wait=f"{wait:3.0f}")
            pbar.close()

            val_conf_norm = val_conf / val_conf.sum(dim=1, keepdim=True).clamp(min=1.0)
            for i, row in enumerate(val_conf_norm):
                print(f"  c{i}: [" + ", ".join([f"{v.item():.2f}" for v in row]) +
                       f"]  recall={row[i].item():.2f}")

            if save_chkp:
                torch.save({'epoch': ep,
                            'model_state_dict': model.module.state_dict()},
                           f"{CHECKPOINT_PATH}/{name}/chkp_{ep:03d}.pt")

        # Broadcast wait so all ranks stop together
        wait_t = torch.tensor(wait, device=device)
        dist.broadcast(wait_t, src=0)
        if wait_t.item() >= patience:
            if IS_MAIN:
                tqdm.write(f"{name} | Early stop at epoch {ep}")
            break

    # Broadcast best weights from rank 0 to all ranks
    for v in best_state.values():
        v_dev = v.to(device)
        dist.broadcast(v_dev, src=0)
    model.module.load_state_dict(
        {k: v.to(device) for k, v in best_state.items()})
    
    dist.barrier()
    if IS_MAIN:
        if save_chkp:
            checkpoint = {'epoch': best_epoch,
            'model_state_dict': model.module.state_dict()}
            if save_chkp: 
                torch.save(checkpoint, f"{CHECKPOINT_PATH}/{name}/{name}.pt")
    return model


@torch.no_grad()
def evaluate_ddp(model, loader, loss_fn,
                 return_emb, return_logits,
                 device, world_size, return_std=False):
    model.eval()
    loss_fn.eval()
    lsum = torch.tensor(0.0, device=device)
    cor = torch.tensor(0.0, device=device)
    tot = torch.tensor(0, device=device, dtype=torch.long)
    val_conf_matrix = torch.zeros((CLASSES, CLASSES), device=device)
    user_conf_matrices = {}  # {user_id: (CLASSES, CLASSES) tensor}

    for xb, yb, ys in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        with autocast(device_type="cuda"):
            if return_emb and return_logits:
                emb, logits = model(xb, return_emb, return_logits)
                loss = loss_fn(emb, logits, yb, ys)
            elif return_emb:
                emb = model(xb, return_emb, return_logits)
                logits = torch.zeros((yb.shape[0], CLASSES)).to(DEVICE)
                loss = loss_fn(emb, yb)
            else:
                logits = model(xb)
                loss = loss_fn(logits, yb)

        preds = logits.argmax(1)

        lsum += loss.detach()
        cor += (preds == yb).sum()
        tot += yb.numel()

        idx = (yb * CLASSES + preds).clamp(0, CLASSES * CLASSES - 1)
        val_conf_matrix += torch.bincount(idx, minlength=CLASSES * CLASSES).float().view(CLASSES, CLASSES)

        for user_id in ys.unique().cpu().tolist():
            mask = ys == user_id
            if user_id not in user_conf_matrices:
                user_conf_matrices[user_id] = torch.zeros((CLASSES, CLASSES), device=device)
            user_idx = (yb[mask] * CLASSES + preds[mask]).clamp(0, CLASSES * CLASSES - 1)
            user_conf_matrices[user_id] += torch.bincount(user_idx, minlength=CLASSES * CLASSES).float().view(CLASSES, CLASSES)

    dist.all_reduce(lsum,            op=dist.ReduceOp.SUM)
    dist.all_reduce(cor,             op=dist.ReduceOp.SUM)
    dist.all_reduce(tot,             op=dist.ReduceOp.SUM)
    dist.all_reduce(val_conf_matrix, op=dist.ReduceOp.SUM)

    user_conf_matrices_list = [None] * world_size
    dist.all_gather_object(user_conf_matrices_list,
                           {uid: m.cpu() for uid, m in user_conf_matrices.items()})
    global_user_conf = {}
    for local_dict in user_conf_matrices_list:
        for user_id, user_conf in local_dict.items():
            if user_id not in global_user_conf:
                global_user_conf[user_id] = torch.zeros((CLASSES, CLASSES), device=device)
            global_user_conf[user_id] += user_conf.to(device)

    user_bal_accs = []
    for user_conf in global_user_conf.values():
        class_totals = user_conf.sum(1)
        recalls = user_conf.diag() / class_totals.clamp(min=1)
        user_bal_accs.append(recalls[class_totals > 0].mean().item())

    bal_acc = np.mean(user_bal_accs) if user_bal_accs else 0.0
    bal_std = np.std(user_bal_accs)  if user_bal_accs else 0.0
    val_conf_matrix = torch.stack(list(global_user_conf.values())).mean(0)

    n_steps = len(loader) * world_size

    if return_std:
        return (cor.item() / max(1, tot.item()),
                lsum.item() / max(1, n_steps),
                bal_acc, val_conf_matrix, bal_std)

    return (cor.item() / max(1, tot.item()),
            lsum.item() / max(1, n_steps),
            bal_acc, val_conf_matrix)


def extract_full(windows, feat_list, feat_dic={}):
    fe = FeatureExtractor()
    return fe.extract_features(feat_list, windows, array=True,
                               fix_feature_errors=True,
                               feature_dic=feat_dic).reshape(windows.shape[0], -1)

def extract_sub(windows, feat_list, feat_dic, n_sub=N_SUB):
    fe = FeatureExtractor()
    N, CH, T = windows.shape
    assert T % n_sub == 0, f"T={T} not divisible by n_sub={n_sub}"
    sub_len = T // n_sub
    # reshape to (N*n_sub, CH, sub_len) so libemg processes all at once
    subs = windows.reshape(N * n_sub, CH, sub_len)
    feats = fe.extract_features(feat_list, subs, array=True,
                                fix_feature_errors=True,
                                feature_dic=feat_dic)
    feats = feats.reshape(N * n_sub, -1)  # (N*n_sub, F)
    return feats.reshape(N, n_sub, -1)    # (N, n_sub, F)


def normalize_features(tr, va, te):
    shape_tr = tr.shape
    shape_va = va.shape
    shape_te = te.shape

    # flatten to 2D for scaler: (N, F) or (N*N_SUB, F)
    tr_2d = tr.reshape(-1, shape_tr[-1])
    va_2d = va.reshape(-1, shape_va[-1])
    te_2d = te.reshape(-1, shape_te[-1])

    scaler = StandardScaler()
    tr_2d  = scaler.fit_transform(tr_2d)   # fit + transform on train
    va_2d  = scaler.transform(va_2d)       # transform only
    te_2d  = scaler.transform(te_2d)       # transform only

    return (np.nan_to_num(tr_2d.reshape(shape_tr), nan=0.0, 
            posinf=0.0, neginf=0.0).astype(np.float32),
            np.nan_to_num(va_2d.reshape(shape_va), nan=0.0, 
            posinf=0.0, neginf=0.0).astype(np.float32),
            np.nan_to_num(te_2d.reshape(shape_te), nan=0.0, 
            posinf=0.0, neginf=0.0).astype(np.float32))


def population_channel_stats(windows, batch=200_000):
    N, C, T = windows.shape
    s1 = np.zeros(C, np.float64)
    s2 = np.zeros(C, np.float64)
    count = 0
    for i in range(0, N, batch):
        c = np.asarray(windows[i:i+batch], dtype=np.float64)
        s1 += c.sum(axis=(0, 2))
        s2 += (c * c).sum(axis=(0, 2))
        count += c.shape[0] * T
    mean = (s1 / count).astype(np.float32)
    std  = np.sqrt(np.clip(s2 / count - (s1/count)**2, 0, None)).astype(np.float32)
    return mean, std


def normalize_per_user(windows, subjects, eps=1e-6):
    out = np.array(windows, dtype=np.float32)  # materialize mmap into writable copy
    
    for s in np.unique(subjects):
        mask = subjects == s
        u = out[mask]                                          # (N_s, C, T)
        mean = u.mean(axis=(0, 2), keepdims=True)              # (1, C, 1)
        std  = u.std(axis=(0, 2), keepdims=True)               # (1, C, 1)
        out[mask] = (u - mean) / (std + eps)
    
    return out

class RunningNorm(nn.Module):
    def __init__(self, num_channels, tau,
                 init_mean=None, init_std=None,
                 eps=1e-6, prior_weight=RN_PRIOR_WEIGHT):   
        
        super().__init__()
        self.tau = tau
        self.eps = eps

        im  = torch.zeros(num_channels) if init_mean is None \
              else torch.as_tensor(init_mean, dtype=torch.float32)
        ist = torch.ones(num_channels)  if init_std  is None \
              else torch.as_tensor(init_std,  dtype=torch.float32)
        im  = im.view(1, -1, 1)
        ist = ist.view(1, -1, 1)

        self.register_buffer("init_mean", im.clone())
        self.register_buffer("init_sq", (ist**2 + im**2).clone())
        self.register_buffer("running_mean", im.clone())
        self.register_buffer("running_sq", (ist**2 + im**2).clone())
        self.register_buffer("n_updates", torch.tensor([float(prior_weight)], dtype=torch.float32))
        self.prior_weight = float(prior_weight)

    @torch.no_grad()
    def _update(self, x):
        B = x.size(0)
        n = self.n_updates.item()
        if self.tau == float('inf'):
            a = B / (n + B)
        else:
            a = 1.0 - math.exp(-B / self.tau)   # simplified: equivalent to 1-(1-alpha)^B

        self.running_mean.mul_(1 - a).add_(
            x.mean(dim=(0, 2), keepdim=True), alpha=a)
        self.running_sq.mul_(1 - a).add_(
            (x * x).mean(dim=(0, 2), keepdim=True), alpha=a)
        self.n_updates.add_(B)

    def forward(self, x):
        if self.training:
            return x
        xf = x.float()
        self._update(xf)
        var = (self.running_sq - self.running_mean ** 2).clamp_min(0.0)
        return ((xf - self.running_mean) / torch.sqrt(var + self.eps) * 128.0).to(x.dtype)

    def reset(self):
        self.running_mean.copy_(self.init_mean)
        self.running_sq.copy_(self.init_sq)
        self.n_updates.fill_(self.prior_weight) 

@torch.no_grad()
def eval_test_running(model, norm_layer, data, name, seed,
                      batch_size=BATCH_SIZE, shuffle=True, warmup=0,
                      device=DEVICE, save=True, csv_path=RESULTS_PATH):
    model.to(device).eval()
    norm_layer.to(device).eval()
    results = {}

    if save:
        os.makedirs(f"{FIGURE_PATH}/{name}/", exist_ok=True)

    for tag, (windows, meta) in data.items():
        subjects = np.asarray(meta['subjects']).reshape(-1)
        labels   = np.asarray(meta['classes']).reshape(-1)
        uniq     = np.unique(subjects)
        n_subj   = len(uniq)
        acc = np.zeros(n_subj); act = np.zeros(n_subj)
        bal = np.zeros(n_subj); f1  = np.zeros(n_subj)
        rng = np.random.default_rng(seed)
        all_preds = np.empty(len(subjects), np.int64)

        for i, s in enumerate(tqdm(uniq, desc=f"{name}/{tag}")):
            norm_layer.reset()
            idx = np.where(subjects == s)[0]
            Xs  = np.asarray(windows[idx], dtype=np.float32)
            ys  = labels[idx]

            if shuffle:
                p   = rng.permutation(len(idx))
                Xs  = Xs[p]; ys = ys[p]

            preds = np.empty(len(idx), np.int64)
            for b in range(0, len(idx), batch_size):
                xb = torch.from_numpy(Xs[b:b+batch_size]).to(device, non_blocking=True)
                xb = norm_layer(xb)
                preds[b:b+xb.size(0)] = model(xb).argmax(1).cpu().numpy()

            # store in original dataset order so all_preds aligns with labels
            if shuffle:
                all_preds[idx[p]] = preds
            else:
                all_preds[idx] = preds

            ps, ls = (preds[warmup:], ys[warmup:]) if warmup else (preds, ys)
            acc[i] = (ps == ls).mean() * 100
            f1[i]  = f1_score(ls, ps, average='macro')
            am     = ls != 0
            act[i] = (ps[am] == ls[am]).mean() * 100 if am.any() else 0.0
            cm     = confusion_matrix(ls, ps, labels=np.arange(CLASSES))
            with np.errstate(divide='ignore', invalid='ignore'):
                bal[i] = np.nanmean(np.diag(cm) / cm.sum(axis=1)) * 100

        print(f"{name}/{tag} | Acc {acc.mean():.2f} ± {acc.std():.2f} "
              f"| Actv {act.mean():.2f} ± {act.std():.2f} "
              f"| Bal {bal.mean():.2f} ± {bal.std():.2f} "
              f"| F1 {f1.mean():.4f} ± {f1.std():.4f}")

        if save:
            _idx = np.argsort(bal)
            fig, axs = plt.subplots(2, 2, figsize=(11, 11), dpi=200)
            ax1, ax2, ax3, ax4 = axs.flatten()
            fig.suptitle(
                f"{tag} | Mean Acc {acc.mean():.2f} ± {acc.std():.2f} "
                f"| Mean Actv {act.mean():.2f} ± {act.std():.2f} "
                f"| Mean Bal {bal.mean():.2f} ± {bal.std():.2f} "
                f"| Mean F1 {f1.mean():.2f} ± {f1.std():.2f}"
            )
            ax1.bar(np.arange(n_subj), acc[_idx]) 
            ax1.axhline(acc.mean(), color='red', linestyle='--')
            ax1.set_title('Per Subject Accuracy')
            ax2.bar(np.arange(n_subj), act[_idx]) 
            ax2.axhline(act.mean(), color='red', linestyle='--')
            ax2.set_title('Per Subject Active Accuracy')
            ax3.bar(np.arange(n_subj), bal[_idx]) 
            ax3.axhline(bal.mean(), color='red', linestyle='--')
            ax3.set_title('Per Subject Balanced Accuracy')
            ax4.bar(np.arange(n_subj), f1[_idx]) 
            ax4.axhline(f1.mean(),  color='red', linestyle='--')
            ax4.set_title('Per F1 Score')
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(f"{FIGURE_PATH}/{name}/{tag}.jpg")
            fig.clf(); plt.close(fig)

            os.makedirs(f"{CHECKPOINT_PATH}/{name}/", exist_ok=True)
            np.save(f"{CHECKPOINT_PATH}/{name}/results_{tag}.npy", np.stack((acc, act, bal, f1)))
            np.save(f"{CHECKPOINT_PATH}/{name}/preds_{tag}.npy",   all_preds)
            np.save(f"{CHECKPOINT_PATH}/{name}/labels_{tag}.npy",  labels)

        results[tag] = {"acc_mean": acc.mean(), "acc_std": acc.std(),
                        "act_acc_mean": act.mean(), "act_acc_std": act.std(),
                        "bal_acc_mean": bal.mean(), "bal_acc_std": bal.std(),
                        "F1": f1.mean(), "F1_std": f1.std()}

    rows = [{"model": name, "test_set": tag, **r} for tag, r in results.items()]
    pd.DataFrame(rows).to_csv(csv_path, mode='a', index=False,
                              header=not os.path.exists(csv_path))

    return results


import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


# ======== AUGMENTATION CONFIG ========
@dataclass
class AugConfig:
    """
    Strength and per sample application probability of each transform.
    All amplitudes are expressed on the raw signed 8 bit ADC scale, i.e.
    the same scale the windows are stored in, before the model divides
    by 128.0 internally.
    """
    # per sample probability that each transform is applied
    p_rotate: float = 0.5
    p_gain: float = 0.5
    p_warp: float = 0.5
    p_noise: float = 0.5

    # electrode rotation: armband donning position around the forearm ring
    rot_max: int = 1            # max shift in electrode positions, plus or minus
    rot_mode: str = 'discrete'  # 'discrete' integer pod shift, 'interp' sub pod blend

    # amplitude gain: per electrode coupling, muscle size, overall effort
    gain_chan: float = 0.25     # per channel half range, log uniform
    gain_global: float = 0.15   # whole sample half range, log uniform

    # magnitude warp: slow non stationary amplitude drift within a window
    warp_sigma: float = 0.15    # std of the warp curve in the log domain
    warp_knots: int = 4         # random knots interpolated up to window length

    # additive noise: sensor noise floor and per session SNR differences
    noise_snr: float = 0.05     # noise std as a fraction of per channel RMS
    noise_floor: float = 1.0    # absolute noise std floor, ADC units

    # keep augmented values inside the signed 8 bit ADC range
    clamp_adc: bool = True
    adc_min: float = -128.0
    adc_max: float = 127.0


# ======== EMG AUGMENTER ========
class EMGAugment(nn.Module):
    """
    On the fly, fully GPU vectorized augmentation for 8 channel Myo EMG.
    Input is a raw ADC scale batch of shape (B, C, T). Output has the
    same shape. Every transform is selected per sample with a boolean
    mask and torch.where, so nothing forces a host device sync and the
    whole batch stays on the GPU.

    Each transform targets a physical source of the cross subject and
    cross session variability that a larger user population provides for
    free, which is what zero shot training is trying to substitute:

      rotate : the band is donned at a different rotation, so each
               electrode sits over a different muscle. On a circular
               electrode ring this is a cyclic shift of the channels.
      gain   : skin electrode impedance, subcutaneous fat, muscle cross
               section and effort all rescale amplitude, differently per
               electrode. This is the user specific amplitude factor the
               encoder otherwise latches onto. Drawn log uniform so a
               1.25x and a 0.8x gain are equally likely.
      warp   : contraction intensity is not stationary across a 200 ms
               window. A smooth positive multiplicative envelope models
               that drift without changing the gesture identity.
      noise  : zero mean Gaussian at a controlled SNR per channel,
               modelling the electrical noise floor and session to
               session SNR changes from sweat, skin prep and aging.

    The transforms compose in donning order: first where the band sits
    (rotate), then the coupling for that placement (gain), then the
    within window effort drift (warp), then the noise floor on top
    (noise).
    """
    def __init__(self, cfg: AugConfig = AugConfig()):
        super().__init__()
        self.cfg = cfg

    # -------- electrode rotation --------
    def _rotate(self, x, m):
        B, C, T = x.shape
        cfg = self.cfg
        if cfg.rot_mode == 'interp':
            # continuous sub pod rotation, linear blend of adjacent channels
            shift = torch.empty(B, device=x.device).uniform_(-cfg.rot_max, cfg.rot_max)
            shift = torch.where(m, shift, torch.zeros_like(shift))
            pos = (torch.arange(C, device=x.device).view(1, C) - shift.view(B, 1)) % C
            lo = torch.floor(pos)
            frac = (pos - lo).view(B, C, 1)
            i0 = (lo.long() % C).unsqueeze(-1).expand(B, C, T)
            i1 = ((lo.long() + 1) % C).unsqueeze(-1).expand(B, C, T)
            return (1.0 - frac) * torch.gather(x, 1, i0) + frac * torch.gather(x, 1, i1)
        # integer pod shift, masked samples get shift 0 and stay unchanged
        shift = torch.randint(-cfg.rot_max, cfg.rot_max + 1, (B,), device=x.device)
        shift = torch.where(m, shift, torch.zeros_like(shift))
        idx = (torch.arange(C, device=x.device).view(1, C) - shift.view(B, 1)) % C
        idx = idx.unsqueeze(-1).expand(B, C, T)
        return torch.gather(x, 1, idx)

    # -------- amplitude gain --------
    def _gain(self, x, m):
        B, C, T = x.shape
        cfg = self.cfg
        lr_c = math.log(1.0 + cfg.gain_chan)
        lr_g = math.log(1.0 + cfg.gain_global)
        g_c = torch.empty(B, C, 1, device=x.device).uniform_(-lr_c, lr_c).exp()
        g_g = torch.empty(B, 1, 1, device=x.device).uniform_(-lr_g, lr_g).exp()
        g = torch.where(m.view(B, 1, 1), g_c * g_g, torch.ones_like(g_c))
        return x * g

    # -------- magnitude warp --------
    def _warp(self, x, m):
        B, C, T = x.shape
        cfg = self.cfg
        knots = torch.randn(B, C, cfg.warp_knots, device=x.device) * cfg.warp_sigma
        curve = F.interpolate(knots, size=T, mode='linear', align_corners=True)
        env = torch.where(m.view(B, 1, 1), curve.exp(), torch.ones_like(curve))
        return x * env

    # -------- additive noise --------
    def _noise(self, x, m):
        B, C, T = x.shape
        cfg = self.cfg
        rms = x.pow(2).mean(dim=2, keepdim=True).clamp_min(1e-8).sqrt()
        sigma = cfg.noise_snr * rms + cfg.noise_floor
        noise = torch.randn_like(x) * sigma
        return x + torch.where(m.view(B, 1, 1), noise, torch.zeros_like(noise))

    @torch.no_grad()
    def forward(self, x):
        cfg = self.cfg
        B = x.shape[0]
        dev = x.device
        x = x.float()
        if cfg.p_rotate > 0:
            x = self._rotate(x, torch.rand(B, device=dev) < cfg.p_rotate)
        if cfg.p_gain > 0:
            x = self._gain(x, torch.rand(B, device=dev) < cfg.p_gain)
        if cfg.p_warp > 0:
            x = self._warp(x, torch.rand(B, device=dev) < cfg.p_warp)
        if cfg.p_noise > 0:
            x = self._noise(x, torch.rand(B, device=dev) < cfg.p_noise)
        if cfg.clamp_adc:
            x = x.clamp(cfg.adc_min, cfg.adc_max)
        return x


# ======== AUGMENTED TRAINING ========
def train_aug(model, train_loader, val_loader, name,
              loss_fn=nn.CrossEntropyLoss(),
              augmenter=None, n_aug=1, keep_clean=True,
              return_emb=False, return_logits=False,
              epochs=EPOCHS, lr=LR_INIT, min_lr=LR_MIN,
              lr_factor=LR_FACTOR, lr_patience=LR_PATIENCE,
              patience=PATIENCE, device=DEVICE,
              verbose=VERBOSE, save_chkp=False):

    model.to(device)
    if augmenter is not None:
        augmenter.to(device)
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr)
    scaler = GradScaler(enabled=(device=="cuda"))

    best_val = 1e9
    best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    wait = 0
    best_epoch = 0

    if save_chkp:
        os.makedirs(f"{CHECKPOINT_PATH}/{name}/", exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0.0, device=device)
        total = 0
        step = 0
        pbar = tqdm(total=len(train_loader), desc=f"{name} | Ep {ep}",
                    leave=True, dynamic_ncols=True, disable=not verbose)

        for xb, yb, *_ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            # -------- build augmented views --------
            if augmenter is not None and (n_aug > 0):
                with torch.no_grad():
                    views = [xb] if keep_clean else []
                    for _ in range(n_aug):
                        views.append(augmenter(xb))
                    xb = torch.cat(views, dim=0)
                    yb = yb.repeat(len(views))

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device=="cuda")):
                if return_emb and return_logits:
                    emb, logits = model(xb, return_emb=True, return_logits=True)
                    loss = loss_fn(emb, logits, yb)
                else:
                    logits = model(xb)
                    loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.detach()
            correct += (logits.argmax(1) == yb).sum()
            total += yb.numel()
            step += 1

            if not (step % UPDATE_EVERY):
                pbar.update(UPDATE_EVERY)
                pbar.set_postfix(
                    loss=f"{total_loss.item() / step:10.8f}",
                    acc=f"{correct.item() / max(1, total):6.4f}",
                    LR=f"{opt.param_groups[0]['lr']:8.6f}")

        if step % UPDATE_EVERY:
            pbar.update(step % UPDATE_EVERY)

        val_acc, val_loss, val_bal, val_conf = evaluate(model, val_loader, loss_fn,
                                                return_emb, return_logits, device)
        sch.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            wait = 0
            best_epoch = ep
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    tqdm.write(f"{name} | Early stop")
                pbar.close()
                break

        pbar.set_postfix(
            loss=f"{total_loss.item() / max(1, len(train_loader)):10.6f}",
            acc=f"{correct.item() / max(1, total):6.4f}",
            val_loss=f"{val_loss:10.6f}",
            val_acc=f"{val_acc:6.4f}",
            val_bal=f"{val_bal:6.4f}",
            LR=f"{opt.param_groups[0]['lr']:8.6f}",
            wait=f"{wait:3.0f}")
        pbar.close()

        if save_chkp:
            checkpoint = {'epoch': ep, 'model_state_dict': model.state_dict()}
            torch.save(checkpoint, f"{CHECKPOINT_PATH}/{name}/chkp_{ep:03d}.pt")

    model.load_state_dict(best_state)
    checkpoint = {'epoch': best_epoch, 'model_state_dict': model.state_dict()}
    if save_chkp:
        torch.save(checkpoint, f"{CHECKPOINT_PATH}/{name}/{name}.pt")
    return model