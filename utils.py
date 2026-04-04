import os, copy, time, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch;
import torch.nn as nn; import torch.nn.functional as F
from torch.optim import Adam
from torch.amp import GradScaler, autocast
from torch.utils.data import (DataLoader, TensorDataset, Sampler)
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
PICKLE_PATH = 'pickles'; CHECKPOINT_PATH = 'checkpoints'; FIGURE_PATH = 'figures'
SEQ = 40; INC = 2; CH = 8; CLASSES = 5; VAL_CUTOFF = 332
WORKERS = 4; PRE_FETCH = 2; VERBOSE=True; DEVICE = 'cuda'
UPDATE_EVERY = 50; PRESIST_WORKER = True; PIN_MEMORY = True

EPOCHS = 200; BATCH_SIZE = 512; DROPOUT = 0.2; PATIENCE = 10
LR_FACTOR = 0.6; LR_PATIENCE = 4; LR_INIT = 1e-4; LR_MIN = 1e-5


# ======== MODELS, TRAINING & DATASETS ========
def count_params(m): 
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ======== DATA LOADER ========
def create_loader(x, y, batch=BATCH_SIZE, shuffle=False, 
                  workers=WORKERS, prefetch_factor=PRE_FETCH,
                  persistent_workers=PRESIST_WORKER,
                  pin_memory=PIN_MEMORY):
    return DataLoader(
    TensorDataset(torch.from_numpy(x), 
                  torch.from_numpy(y)),
                #   torch.tensor(x), 
                #   torch.tensor(y)),
    batch_size=batch,
    shuffle=shuffle,
    num_workers=workers,
    prefetch_factor=prefetch_factor if workers > 0 else None,
    persistent_workers=persistent_workers,
    pin_memory=pin_memory,
    drop_last=False)


def create_loader_grl(x, y, s, batch=BATCH_SIZE, shuffle=False, 
                  workers=WORKERS, prefetch_factor=PRE_FETCH,
                  persistent_workers=PRESIST_WORKER,
                  pin_memory=PIN_MEMORY):
    return DataLoader(
    TensorDataset(torch.from_numpy(x), 
                  torch.from_numpy(y),
                  torch.from_numpy(s)),
                #   torch.tensor(x), 
                #   torch.tensor(y)),
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
        self.n_samples = max(2, int(batch_size) // (self.n_classes * self.n_subjects))
        self.batch_size = self.n_classes * self.n_subjects * self.n_samples

        # Remap subjects to 0..S-1 (handles train 0..305 and test 305..611 cleanly)
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
        # Use a local generator derived from global seed + epoch (no global RNG reset)
        g = torch.Generator()
        g.manual_seed(torch.initial_seed() + self.seed_offset + self.epoch)

        # Shuffle subjects and iterate in blocks of n_subjects (strict diversity)
        subj_perm = torch.randperm(self.S, generator=g)
        subj_ptr = 0

        # Per-cell cursor for sequential consumption (no toss)
        K = self.n_classes * self.S
        cursor = torch.zeros(K, dtype=torch.long)

        def take_from_cell(cell: int) -> torch.Tensor:
            cnt = int(self.counts[cell].item())
            if cnt <= 0:
                # should not happen with your guarantee; fallback is global random
                ridx = torch.randint(0, self.order.numel(), (self.n_samples,), generator=g)
                return self.order[ridx]

            p = int(cursor[cell].item())

            # sequential without replacement
            if p + self.n_samples <= cnt:
                lo = int(self.starts[cell].item()) + p
                hi = lo + self.n_samples
                out = self.order[lo:hi]
                cursor[cell] = p + self.n_samples
                return out

            # exhausted: reuse (no toss)
            if self.reuse_mode == "replacement":
                # sample within-cell with replacement
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
                # cnt < n_samples: pad within-cell (replacement)
                lo0 = int(self.starts[cell].item())
                ridx = torch.randint(0, cnt, (self.n_samples - out.numel(),), generator=g)
                pad = self.order[lo0 + ridx]
                out = torch.cat([out, pad], dim=0)

            cursor[cell] = 0
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

    if save_chkp:
        os.makedirs(f"{CHECKPOINT_PATH}", exist_ok=True)
        os.makedirs(f"{CHECKPOINT_PATH}/{name}/", exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0.0, device=device)
        total = 0
        step = 0
        pbar = tqdm(total=len(train_loader), desc=f"{name} | Ep {ep}", 
                    leave=True, dynamic_ncols=True, disable=not verbose)

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device=="cuda")):
                if return_emb and return_logits:
                    emb, logits = model(xb, return_emb, return_logits)
                    loss = loss_fn(emb, logits, yb)
                else:
                    logits = model(xb)
                    loss = loss_fn(logits, yb)                    

            scaler.scale(loss).backward()
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

        val_acc, val_loss = evaluate(model, val_loader, loss_fn, 
                                     return_emb, return_logits, device)
        sch.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            wait = 0
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
            LR=f"{opt.param_groups[0]['lr']:8.6f}",
            wait=f"{wait:3.0f}")
        pbar.close()

        if save_chkp:
            checkpoint = {'epoch': ep,
                        'model_state_dict': model.state_dict()}
            torch.save(checkpoint, f"{CHECKPOINT_PATH}/{name}/chkp_{ep:03d}.pt")

    model.load_state_dict(best_state)
    return model


def train_grl(model, train_loader, val_loader, name,
            loss_fn=nn.CrossEntropyLoss(),
            loss_fn_grl=None,
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

    if save_chkp:
        os.makedirs(f"{CHECKPOINT_PATH}", exist_ok=True)
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

        for xb, yb, ysb in train_loader:
            p = global_steps / max_steps
            lmbd = np.clip(p, 0, 1)
            model.grl.lambd = float(lmbd)

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            ysb = ysb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device=="cuda")):
                logits, logits_grl = model(xb, return_grl=True)
                loss_c = loss_fn(logits, yb)
                loss_grl = loss_fn_grl(logits_grl, ysb)
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

        val_acc, val_loss = evaluate(model, val_loader, loss_fn, 
                                     return_emb, return_logits, device)
        sch.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            wait = 0
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
            LR=f"{opt.param_groups[0]['lr']:8.6f}",
            wait=f"{wait:3.0f}")
        pbar.close()

        if save_chkp:
            checkpoint = {'epoch': ep,
                        'model_state_dict': model.state_dict()}
            torch.save(checkpoint, f"{CHECKPOINT_PATH}/{name}/chkp_{ep:03d}.pt")

    model.load_state_dict(best_state)
    return model


def train_triplet(model, train_loader, val_loader, name,
          epochs=EPOCHS, lr=LR_INIT, min_lr=LR_MIN,
          criterion_ce=nn.CrossEntropyLoss(), criterion_tri=None,
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

    if save_chkp:
        os.makedirs(f"{CHECKPOINT_PATH}/{name}/", exist_ok=True)

    ep = 1
    while ep <= epochs:
        progress = min((ep - 1) / warmup_epochs, 1.0)
        current_alpha = alpha_start - (alpha_start - alpha_end) * progress
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

        for xb, yb, sb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            sb = sb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device=="cuda")):
                emb, logits = model(xb, return_emb=True, return_logits=True)
                
                loss_ce = criterion_ce(logits, yb)
                loss_tri = criterion_tri(emb, yb, sb)
                
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

        val_acc, val_loss_ce, val_loss_tri = evaluate_triplet(
            model, val_loader, criterion_ce, device, 
            triplet_fn=criterion_tri, alpha=current_alpha)
        
        monitor_metric = (current_ce_w * val_loss_ce) + (current_alpha * val_loss_tri)
        sch.step(monitor_metric)

        if monitor_metric < best_val_metric:
            best_val_metric = monitor_metric
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            wait = 0
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
            LR=f"{opt.param_groups[0]['lr']:.6f}",
            wait=f"{wait:2d}")
        pbar.close()

        if save_chkp:
            checkpoint = {'epoch': ep,
                          'model_state_dict': model.state_dict()}
            torch.save(checkpoint, f"{CHECKPOINT_PATH}/{name}/chkp_{ep:03d}.pt")

        ep += 1

    model.load_state_dict(best_state)
    return model


# ---- EMBEDDING PCA CALLBACK ----
class PCA_GPU:
    def __init__(self, dims=2, device=DEVICE):
        self.device = device
        self.dims = dims
        self.mean_ = None
        self.components_ = None

    def fit(self, x):
        if isinstance(x, np.ndarray):
            X = torch.from_numpy(x).to(self.device)
        else:
            X = x.to(self.device)
        N = X.shape[0]
        self.mean_ = X.mean(dim=0, keepdim=True)
        Xc = X - self.mean_
        C = (Xc.T @ Xc) / (N - 1)
        eigvals, eigvecs = torch.linalg.eigh(C)
        idx = torch.argsort(eigvals, descending=True)
        self.components_ = eigvecs[:, idx[:self.dims]]
        return self

    def transform(self, x):
        if isinstance(x, np.ndarray):
            X = torch.from_numpy(x).to(self.device)
        else:
            X = x.to(self.device)
        Xc = X - self.mean_
        Z = Xc @ self.components_
        return Z

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

@torch.no_grad()
def collect_embeddings(model, loader, device):
    model.eval()
    N = len(loader.dataset)
    
    # Robustly infer embedding dimension
    sample_xb, _ = next(iter(loader))
    with autocast(device_type="cuda", enabled=(device=="cuda")):
        sample_emb = model(sample_xb.to(device), return_emb=True)
    D = sample_emb.shape[1]
    
    feats = torch.empty(N, D, device=device)
    labels = torch.empty(N, device=device, dtype=torch.long)
    
    ptr = 0
    for xb, yb in loader:
        b = xb.size(0)
        xb = xb.to(device, non_blocking=True)
        
        # Match training precision for speed
        with autocast(device_type="cuda", enabled=(device=="cuda")):
            emb = model(xb, return_emb=True)
            
        feats[ptr:ptr+b] = emb
        labels[ptr:ptr+b] = yb.to(device, non_blocking=True)
        ptr += b
        
    return feats, labels # Return on GPU for PCA fitting

@torch.no_grad()
def run_pca_sweep(model, loader, name, device=DEVICE):
    checkpoint_dir = f"{CHECKPOINT_PATH}/{name}/"
    output_dir = f"{FIGURE_PATH}/{name}_PCAs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all checkpoint files and sort by epoch
    files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
    if not files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    model.to(device)
    model.eval()
    
    # 1. Collect all embeddings for all epochs first (to fit a global PCA for stable video/plots)
    # If memory is an issue, fit PCA only on the last epoch's embeddings
    all_epoch_data = []
    
    for f in files:
        ep_path = os.path.join(checkpoint_dir, f)
        checkpoint = torch.load(ep_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        
        feats, labels = collect_embeddings(model, loader, device)
        all_epoch_data.append({
            "epoch": epoch,
            "feats": feats,  # These are already on CPU from collect_embeddings
            "labels": labels
        })

    # 2. Fit PCA on the final epoch to define the coordinate space
    # (Using the last epoch ensures the most discriminative features define the axes)
    pca = PCA_GPU(dims=2, device=device)
    pca.fit(all_epoch_data[-1]["feats"])

    # 3. Transform and Plot
    for data in all_epoch_data:
        ep = data["epoch"]
        Z = pca.transform(data["feats"]).cpu().numpy()
        y = data["labels"].numpy()

        # Fast plotting using OO API
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100) # Lower DPI for faster saving
        scatter = ax.scatter(Z[:, 0], Z[:, 1], c=y, s=4, cmap="tab10", alpha=0.6)
        ax.set_title(f"{name} | Epoch {ep}")
        
        # Hide axes for cleaner visualization if preferred
        ax.set_xticks([]); ax.set_yticks([])
        
        fig.savefig(f"{output_dir}/ep_{ep:03d}.png", bbox_inches='tight')
        plt.close(fig) # Mandatory memory release

    print(f"PCA sweep completed. Figures saved to {output_dir}")

# ---- VALIDATION ----
@torch.no_grad()
def evaluate(model, loader, loss_fn, 
             return_emb, return_logits, device):
    model.eval()
    # Initialize on GPU
    lsum = torch.tensor(0.0, device=device)
    cor = torch.tensor(0.0, device=device)
    tot = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
            if return_emb and return_logits:
                emb, logits = model(xb, return_emb, return_logits)
                loss = loss_fn(emb, logits, yb)
            else:
                logits = model(xb)
                loss = loss_fn(logits, yb)  
        lsum += loss.detach()
        cor += (logits.argmax(1) == yb).sum()
        tot += yb.numel()
    return cor.item() / max(1, tot), lsum.item() / max(1, len(loader))


@torch.no_grad()
def evaluate_triplet(model, loader, loss_fn, 
            device, triplet_fn=None, alpha=0.0):
    model.eval()
    lsum = torch.tensor(0.0, device=device)
    cor = torch.tensor(0.0, device=device)
    tri_sum = torch.tensor(0.0, device=device)
    tot = 0
    
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
    
    return acc, avg_ce, avg_tri


# ======== GENERAL TESTING ========
@torch.no_grad()
def eval_test(model, loaders, metas, name,
              save=True, multi_head=None,
              device=DEVICE):

    model.to(device)
    model.eval()
    results = {}
    os.makedirs(f"{FIGURE_PATH}", exist_ok=True)
    os.makedirs(f"{FIGURE_PATH}/{name}/", exist_ok=True)

    def run(loader, meta, tag):
        N = len(loader.dataset)
        # Pre-allocate on GPU to avoid dynamic growth
        preds = torch.empty(N, dtype=torch.long, device=device)
        ptr = 0
        for xb, _ in loader:
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
        
        acc, act_acc, bal_acc = np.zeros(n_subj), np.zeros(n_subj), np.zeros(n_subj)

        for i, s in enumerate(unique_subjects):
            mask = (subjects == s)
            ps, ls = preds[mask], labels[mask]

            # CA (Classification Accuracy)
            acc[i] = (ps == ls).mean()

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

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 7), dpi=200)
        fig.suptitle(f"{tag} | Mean Acc {acc.mean():.2f} ± {np.std(acc):.2f} \
                     | Mean Actv {act_acc.mean():.2f} ± {np.std(act_acc):.2f}")
        
        ax1.bar(np.arange(n_subj), np.sort(acc))
        ax1.axhline(acc.mean(), color='red', linestyle='--')
        ax1.set_title('Per Subject Accuracy')
        
        ax2.bar(np.arange(n_subj), np.sort(act_acc))
        ax2.axhline(act_acc.mean(), color='red', linestyle='--')
        ax2.set_title('Per Subject Active Accuracy')

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f"{FIGURE_PATH}/{name}/{tag}.jpg")
        fig.clf()
        plt.close(fig)

        if save:
            os.makedirs(f"{CHECKPOINT_PATH}", exist_ok=True)
            os.makedirs(f"{CHECKPOINT_PATH}/{name}/", exist_ok=True)
            np.save(f"{CHECKPOINT_PATH}/{name}/acc_{tag}.npy", acc)
            np.save(f"{CHECKPOINT_PATH}/{name}/aer_{tag}.npy", act_acc)

        return {"acc_mean": acc.mean(), "acc_std": acc.std(),
                "act_acc_mean": act_acc.mean(), "act_acc_std": act_acc.std(),
                "bal_acc_mean": bal_acc.mean(), "bal_acc_std": bal_acc.std()}

    # Iterate through provided loaders (raw, segmented, relabeled)
    for tag in loaders.keys():
        results[tag] = run(loaders[tag], metas[tag], tag)

    # Atomic CSV logging (Fastest concurrent-safe method)
    csv_path = f"{FIGURE_PATH}/results.csv"
    rows = [{"model": name, "test set": tag, **r} for tag, r in results.items()]
    empty_row = {k: "" for k in rows[0].keys()}
    rows.insert(0, empty_row)
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
        for xb, _ in loader:
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


def eval_test_lda(model, X, metas, name, save=True):

    results = {}
    os.makedirs(f"{FIGURE_PATH}", exist_ok=True)
    os.makedirs(f"{FIGURE_PATH}/{name}/", exist_ok=True)

    def run(_x, meta, tag):
        preds = model.predict(_x)
        
        subjects = np.asarray(meta['subjects'])
        labels = np.asarray(meta['classes'])
        unique_subjects = np.unique(subjects)
        n_subj = len(unique_subjects)
        
        acc, act_acc, bal_acc = np.zeros(n_subj), np.zeros(n_subj), np.zeros(n_subj)

        for i, s in enumerate(unique_subjects):
            mask = (subjects == s)
            ps, ls = preds[mask], labels[mask]

            # CA (Classification Accuracy)
            acc[i] = (ps == ls).mean()

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

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 7), dpi=200)
        fig.suptitle(f"{tag} | Mean Acc {acc.mean():.2f} ± {np.std(acc):.2f} \
                     | Mean Actv {act_acc.mean():.2f} ± {np.std(act_acc):.2f}")
        
        ax1.bar(np.arange(n_subj), np.sort(acc))
        ax1.axhline(acc.mean(), color='red', linestyle='--')
        ax1.set_title('Per Subject Accuracy')
        
        ax2.bar(np.arange(n_subj), np.sort(act_acc))
        ax2.axhline(act_acc.mean(), color='red', linestyle='--')
        ax2.set_title('Per Subject Active Accuracy')

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f"{FIGURE_PATH}/{name}/{tag}.jpg")
        fig.clf()
        plt.close(fig)

        if save:
            os.makedirs(f"{CHECKPOINT_PATH}", exist_ok=True)
            os.makedirs(f"{CHECKPOINT_PATH}/{name}/", exist_ok=True)
            np.save(f"{CHECKPOINT_PATH}/{name}/acc_{tag}.npy", acc)
            np.save(f"{CHECKPOINT_PATH}/{name}/aer_{tag}.npy", act_acc)

        return {"acc_mean": acc.mean(), "acc_std": acc.std(),
                "act_acc_mean": act_acc.mean(), "act_acc_std": act_acc.std(),
                "bal_acc_mean": bal_acc.mean(), "bal_acc_std": bal_acc.std()}

    # Iterate through provided loaders (raw, segmented, relabeled)
    for tag in X.keys():
        results[tag] = run(X[tag], metas[tag], tag)

    # Atomic CSV logging (Fastest concurrent-safe method)
    csv_path = f"{FIGURE_PATH}/results.csv"
    rows = [{"model": name, "test_set": tag, **r} for tag, r in results.items()]
    empty_row = {k: "" for k in rows[0].keys()}
    rows.insert(0, empty_row)
    df_new = pd.DataFrame(rows)
    df_new.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))

    return results