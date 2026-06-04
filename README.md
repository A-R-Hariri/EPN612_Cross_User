# Cross-User Zero-Shot EMG Gesture Classification on EPN612

End-to-end model design and benchmark for zero-shot cross-user EMG gesture classification. Models are trained on a population of users and evaluated on entirely held-out subjects with no per-user calibration data. This is the research codebase for the MSc thesis *"Enabling User-Invariant Calibration-Free Myoelectric Control for Scalable Human-Machine Interaction"* (University of New Brunswick, IBME).

---

## Problem

Standard EMG classifiers require a per-user calibration session because surface EMG varies substantially across individuals in amplitude, spatial distribution, and spectral content. This work asks whether a single classifier trained on a large population can generalize to new users with zero enrollment data.

The primary objective is not mean accuracy alone. A model with ~90% overall where some users hit 100% and others 30% is considered inferior to one with ~80% where all users fall between 60–90%. Per-user accuracy consistency (reported as mean and standard deviation of per-subject balanced accuracy across 280 test users) is the central optimization target.

---

## Dataset: EMG-EPN-612

- **612 subjects**, Myo armband (8-channel sEMG, 200 Hz, signed 8-bit ADC)
- **5 gesture classes**: NM (rest/no motion), HC (hand close/fist), FX (flexion/waveIn), EX (extension/waveOut), HO (hand open)
- **Original format**: per-user JSON files with training repetitions (50 reps/gesture) and testing repetitions (25 reps/gesture), with ground-truth activity boundaries

Place the raw dataset at `EPN612/` in the repo root. The dataset is available from the original EPN-612 paper authors.

**Fixed user split — never shuffled:**

| Split | Users (1-indexed) | Count | Role |
|-------|-------------------|-------|------|
| Train | 1–306 | 306 | Model training |
| Validation | 307–332 | 26 | Hyperparameter tuning, early stopping |
| Test | 333–612 | 280 | Zero-shot evaluation on unseen users |

**Input scaling**: all models divide raw EMG by `128.0` internally. This is bit-depth normalization for the signed 8-bit Myo ADC — it is not a per-user or per-window normalization step.

---

## Repository Structure

```
EPN612_Cross_User/
    EPN612.py           -- LibEMG Dataset class; JSON-to-HDF5 converter
    process_epn612.py   -- Full preprocessing pipeline (4 variants: raw, segmented, relabeled, standard)
    models.py           -- All model architectures and loss functions
    utils.py            -- Hyperparameters, loaders, training loops, evaluation, normalization utilities

    cross_feats.py      -- Feature grid search: 11 feature groups x LDA/MLP/LSTM_HCF/CNN_HCF (DDP)
    cross_models.py     -- Architecture comparison: LDA, MLP, CNN, MHCNN, LSTM, LSTM_HCF, CNN_HCF (DDP)
    cross_mhcnn.py      -- Proposed model + all loss variants; RunningNorm evaluation (single GPU)

    within_mhcnn.py     -- Within-user MHCNN across all 612 subjects and rep-count sweep
    within_cnnhcf.py    -- Within-user CNN_HCF on hand-crafted sub-windowed features
    within_mlp.py       -- Within-user MLP on hand-crafted sub-windowed features
    within_lda.py       -- Within-user LDA on hand-crafted sub-windowed features

    inc_mhcnn.py        -- Incremental training: subjects x reps sweep with user-ordering strategies

    Analysis_PCA.py     -- PCA projection of the embedding space from each epoch's checkpoint
```

---

## Dependencies

```
torch
numpy
scipy
scikit-learn
matplotlib
pandas
h5py
libemg
tqdm
filelock
```

```bash
pip install torch numpy scipy scikit-learn matplotlib pandas h5py libemg tqdm filelock
```

DDP training (`cross_feats.py`, `cross_models.py`) requires `torchrun`.

---

## Data Pipeline

### Step 1 — JSON to HDF5

`EPN612.py` defines `EMGEPN612`, a LibEMG-compatible Dataset class. On first call to `prepare_data()`, it converts the per-user JSON files to per-user HDF5 files under `EPN612_PROCESSED/`. Each file stores repetitions with raw EMG, gesture label, subject ID, and ground-truth activity boundaries (`pb`, `pe`) derived from the JSON `groundTruth` field.

### Step 2 — Window extraction (`process_epn612.py`)

Run once before any training:

```bash
python process_epn612.py
```

This produces four preprocessing variants, all saved to `pickles/`. Windows are shape `(N, 8, 40)` — 8 channels, 40 samples (200 ms at 200 Hz), stride 2 samples (10 ms). Rest windows are subsampled 4:1 after windowing in all variants to reduce class imbalance.

**Raw** (`*_raw`): windows extracted from full repetition sequences with no temporal segmentation.

**Segmented** (`*_segmented`): an active-segment detector is applied to each active-gesture repetition using signal energy (squared amplitude on the highest-energy channel, smoothed). Frames outside the detected active window are discarded before windowing. Rest repetitions are left intact.

**Relabeled** (`*_relabeled`): uses segmentation boundaries to relabel pre- and post-gesture transition frames as rest (class 0) rather than discarding them, creating a larger and more realistic rest distribution.

**Standard** (`*_standard`): uses a population-level amplitude threshold (mean + 3σ from training rest windows) applied via a sliding-window energy detector. The threshold is computed from training data only and applied uniformly to all splits.

Each variant produces six `.npy` files:
```
pickles/
    {train,val,test}_windows_{variant}.npy   -- (N, 8, 40) float32
    {train,val,test}_meta_{variant}.npy      -- dict: classes, subjects, reps, base_class
```

---

## Models

All models are defined in `models.py`. Raw-input models apply `x / 128.0` internally.

**MHCNN (Multi-Horizon CNN, proposed)**: three parallel dilated Conv1d branches (kernel=8, dilations 1/2/4) on the raw 8-channel input, capturing activation dynamics at approximately 40 ms, 80 ms, and 160 ms receptive fields simultaneously. Branch outputs are concatenated and passed through a final Conv1d (kernel=4), AdaptiveAvgPool1d, a 128-unit GELU FC layer, and a 128-dimensional embedding head. The gesture classifier is a linear layer on the embedding.

**MHCNN_GRL**: MHCNN with a gradient reversal layer (GRL) and a secondary 306-class user-identity classifier. Used for domain-adversarial training (DANN).

**CNN**: single-scale sequential Conv1d (3 layers, kernel sizes 4/3/3) — ablation of MHCNN that isolates the contribution of multi-horizon parallel dilation.

**MLP**: 3-layer fully connected network on hand-crafted LibEMG features (single window, no temporal context).

**LSTM**: 3-layer bidirectional LSTM on raw EMG timesteps (40 steps × 8 channels).

**LSTM_HCF**: 3-layer LSTM on sub-windowed hand-crafted features. The 40-sample window is split into 4 sub-windows of 10 samples each; features are extracted per sub-window, producing a (4, F) temporal sequence.

**CNN_HCF**: 1D CNN on sub-windowed features. Input shape is (B, F, 4) — F features as channels, 4 sub-windows as the temporal dimension.

---

## Loss Functions

All loss functions accept `(logits, labels, *args)` so they are interchangeable in the training loops.

| Key | Class | Description |
|-----|-------|-------------|
| `base` | `BaseLoss` | Standard cross-entropy |
| `rest` | `RestLoss` | CE + additive penalty for misclassifying rest as active or active as wrong gesture |
| `act` | `ActiveLoss` | Upweights loss on active gesture samples |
| `cvar` | `CVaRLoss` | Conditional Value at Risk: CE on the top-α% hardest samples per batch |
| `std` | `STDLoss` | CE + regularization penalizing inter-subject loss variance directly |
| `sbj` | `PerSubjectLoss` | Per-subject loss equity term |
| `proto` | `PrototypeLoss` | CE + within-class prototype compactness |
| `1va` | `OneVsAllLoss` | CE + prototypical CE over negative squared distances to class means |
| `ang` | `AngularLoss` | Direction-based contrastive loss respecting the radial embedding geometry |
| `grl` | — | MHCNN_GRL + `BaseLoss` + domain classifier CE via gradient reversal |
| `trp` | `TripletLoss` | CE + batch-hard triplet loss with cross-user positives and same-user negatives |

---

## Training Scripts

### `cross_feats.py` — Feature grid search (DDP)

Trains LDA, MLP, LSTM_HCF, and CNN_HCF on 11 predefined feature groups across the cross-user split.

```bash
torchrun --nproc_per_node=<N> cross_feats.py <GPU>
```

Feature groups include: WENG, RMS, HTD (MAV/ZC/SSC/WL), DFTR, ITD, LS4, TDAR, COMB, MSWT, TDPSD, LS9.

### `cross_models.py` — Architecture comparison (DDP)

Trains all architectures (LDA, MLP, LSTM_HCF, CNN_HCF, LSTM, CNN, MHCNN) on raw windows under the same conditions.

```bash
torchrun --nproc_per_node=<N> cross_models.py <GPU>
```

### `cross_mhcnn.py` — Proposed model + loss benchmark

Trains the MHCNN under all loss variants and evaluates across all four preprocessing variants. Supports a `RunningNorm` evaluation path.

```bash
python cross_mhcnn.py <GPU> <TAG> <norm|nonorm> <variant[,variant,...]|all> <train|eval>
```

Examples:
```bash
# Train all loss variants on raw windows
python cross_mhcnn.py 0 raw nonorm all train

# Train only base and triplet
python cross_mhcnn.py 0 raw nonorm base,trp train

# Evaluate with RunningNorm (streaming EMA normalization)
python cross_mhcnn.py 0 raw norm base eval
```

**`RunningNorm`** is a streaming per-channel EMA normalization module initialized from population statistics (mean and std computed from training windows). It adapts per-channel input distributions at inference time without any labeled calibration data. Time constant `tau` controls the EMA window: `tau=inf` is exact cumulative mean; finite `tau` is exponential moving average.

### `within_mhcnn.py` — Within-user upper bound

Trains a separate MHCNN per subject across all 612 users (train, val, and test splits), sweeping over rep counts.

```bash
python within_mhcnn.py <GPU> <TAG> <rep[,rep,...]> <ft|noft>
```

The `ft` flag enables fine-tuning from a pretrained cross-user checkpoint rather than training from scratch. Reps are drawn from the first N repetitions of each gesture; rep 14 is always held for validation, rep 15–19 for test.

### `within_cnnhcf.py` — Within-user CNN_HCF

Same structure as `within_mhcnn.py` but operating on sub-windowed hand-crafted features.

### `inc_mhcnn.py` — Incremental training

Sweeps the MHCNN over a grid of training-user counts (1, 2, 4, 8, 16, 32, 64, 128, 196, 306) and rep counts (1, 2, 4, 8, 16, 24, 32, 40, 50) with three user-ordering strategies: worst-first (ranked by lowest within-user balanced accuracy), best-first (ranked highest first), and random (seeded shuffle).

```bash
python inc_mhcnn.py <GPU> <TAG> <seed_run[,...]> <loss_variant>
```

Seed run 0 = worst-first ordering, 1 = best-first, 2+ = random shuffles with deterministic seeds.

---

## Evaluation

All evaluation is performed per subject. For each test user, four metrics are computed independently and then aggregated across the 280 test users:

- **Accuracy**: fraction of windows correctly classified
- **Active accuracy**: accuracy restricted to non-rest windows only
- **Balanced accuracy**: macro-averaged per-class recall (accounts for the rest-class imbalance)
- **Macro F1**: unweighted average of per-class F1

Results are reported as mean ± std across subjects. The per-subject balanced accuracy distribution, sorted by accuracy, is plotted and saved for each run. Aggregated results are appended to `figures/results.csv`.

The primary benchmark metric is **per-subject balanced accuracy mean ± std** on the 280-user test set. Low variance is treated as equal in importance to high mean.

Evaluation always uses raw test windows (`test_windows_raw`) as the primary split, with the trained model also evaluated on segmented, relabeled, and standard splits to quantify sensitivity to the preprocessing variant.

---

## `Analysis_PCA.py`

Loads all per-epoch checkpoints saved during a `cross_mhcnn.py` run, fits PCA on the final-epoch embeddings of the test set, and renders 2D scatter plots per epoch colored by gesture class. Plots are saved to `figures/<name>_PCAs_2/`. This allows inspection of how the embedding space evolves over training.

---

## Output Layout

```
pickles/          -- Preprocessed windows and metadata (.npy)
checkpoints/      -- Per-epoch and best-epoch model weights (.pt); per-subject results (.npy)
figures/          -- Per-subject accuracy bar charts; PCA plots
figures/results.csv -- Aggregated accuracy summary across all runs
```

---

## Key Hyperparameters (from `utils.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SEQ` | 40 | Window length (samples, 200 ms) |
| `INC` | 2 | Window stride (samples, 10 ms) |
| `CH` | 8 | EMG channels |
| `CLASSES` | 5 | Gesture classes |
| `VAL_CUTOFF` | 332 | 0-indexed user ID boundary for val/test split |
| `BATCH_SIZE` | 512 | Training batch size |
| `EPOCHS` | 100 | Maximum training epochs |
| `PATIENCE` | 5 | Early stopping patience (validation loss) |
| `LR_INIT` | 1e-4 | Initial learning rate (Adam) |
| `DROPOUT` | 0.2 | Dropout rate |

---

## Author

Amir Hariri, Institute of Biomedical Engineering, University of New Brunswick.

---

## License

MIT — see [LICENSE](LICENSE).
