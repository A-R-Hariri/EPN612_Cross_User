# Cross-User Zero-Shot EMG Gesture Classification on EMG-EPN-612

Research codebase for zero-shot cross-user surface EMG gesture classification. Models are trained on one population of users and evaluated on a disjoint set of users with no calibration data, no fine-tuning, and no per-user statistics of any kind at inference time.

The code supports the MSc thesis work at the University of New Brunswick Institute of Biomedical Engineering on user-invariant, calibration-free myoelectric control, and the offline half of the associated Fitts' law online study.

## Objective

Conventional myoelectric classifiers require a per-user screen-guided training session because surface EMG differs across users in amplitude, electrode-to-muscle alignment, subcutaneous fat, and spectral content. The question here is whether a single fixed model trained on hundreds of users transfers to a new user with zero enrollment data.

Mean accuracy alone is not the target. A model at 90 percent mean accuracy where some users reach 100 percent and others fall to 60 percent is treated as worse than a model at 82 percent where every user lands between 75 and 90 percent. For a mouse-control or prosthesis application, a user in the low tail has an unusable device regardless of the population mean. The reported benchmark metric is therefore per-subject balanced accuracy across the 280 test users, mean and standard deviation together, and the spread is weighted as heavily as the centre.

## Dataset

EMG-EPN-612, collected with the Myo armband: 8 channels, 200 Hz, signed 8-bit ADC.

Five classes are used: NM (rest, no motion), HC (hand close, fist), FX (flexion, waveIn), EX (extension, waveOut), HO (hand open). The dataset also contains a pinch gesture (class 5); it is dropped during preprocessing.

Rest is heavily over-represented after windowing, so rest windows are subsampled 4:1 in every preprocessing variant. Balanced accuracy is the primary metric for the same reason.

The raw dataset goes in `EPN612/` at the repository root. It is distributed by the original EPN-612 authors.

### Fixed split

The split is fixed and never shuffled, so results stay comparable with prior EPN-612 work and across runs of this repository.

| Split | User IDs (1-indexed) | Subject IDs in code (0-indexed) | Count | Role |
|---|---|---|---|---|
| Train | 1 to 306 | 0 to 305 | 306 | Training |
| Validation | 307 to 332 | 306 to 331 | 26 | Early stopping, LR schedule, model selection |
| Test | 333 to 612 | 332 to 611 | 280 | Zero-shot evaluation |

`VAL_CUTOFF = 332` is the 0-indexed boundary between validation and test. Per-user results arrays are indexed by `subject_rel = subject - VAL_CUTOFF`.

Seeds affect weight initialisation and batch order only. Monte Carlo resampling of the user partition across six seeds produced between-partition variance below the finite-subject sampling error, so the canonical split is used throughout rather than K-fold.

### Input scaling

Every raw-input model divides its input by `128.0` inside `forward`. That is bit-depth scaling for the signed 8-bit Myo ADC, not normalisation. The network sees the untouched EMG waveform.

Per-window and per-channel z-score normalisation were both tested and underperformed raw input in this encoder. Per-user normalisation is not applicable, since any per-user statistic breaks the zero-shot premise. The only adaptive scheme retained is `RunningNorm`, which estimates channel statistics online from unlabelled streaming data and is documented below.

## Repository layout

```
EPN612.py             LibEMG Dataset class for EMG-EPN-612; JSON to HDF5 conversion
process_epn612.py     Windowing pipeline; produces the four preprocessing variants
models.py             Architectures and loss functions
utils.py              Hyperparameters, loaders, samplers, training loops, evaluation,
                      RunningNorm, feature extraction, GPU augmentation

cross_feats.py        Feature-set grid search, 11 groups x 4 shallow models (DDP)
cross_models.py       Architecture comparison on the fixed split (DDP)
cross_mhcnn.py        Proposed model, full loss benchmark, RunningNorm evaluation
inc_mhcnn.py          Users x repetitions data-scaling sweep
inc_mhcnn_aug.py      Same sweep with on-device augmentation enabled

within_mhcnn.py       Within-user MHCNN, all 612 users, repetition sweep, optional fine-tune
within_cnnhcf.py      Within-user CNN on sub-windowed features
within_lstmhcf.py     Within-user LSTM on sub-windowed features
within_mlp.py         Within-user MLP on full-window features
within_lda.py         Within-user LDA baseline

Analysis_PCA.py       Embedding-space PCA projections from saved checkpoints
Analysis_Taxonomy.py  Per-user failure-mode taxonomy (ASI and RLI) and population statistics
eval_sgt_cross.py     Applies EPN-trained cross-user models to online-study participants' SGT data
```

## Requirements

```bash
pip install torch numpy scipy scikit-learn pandas matplotlib h5py libemg tqdm statsmodels
```

`cross_feats.py` and `cross_models.py` run under `torchrun`. `Analysis_PCA.py` additionally imports `torchview` and `cairosvg` for the architecture graph.

## Data pipeline

### Step 1: JSON to HDF5

`EPN612.py` defines `EMGEPN612`, a LibEMG-compatible `Dataset`. The first call to `prepare_data()` converts the per-user JSON files into per-user HDF5 files under `EPN612_PROCESSED/`. Each repetition stores raw EMG, gesture label, subject ID, repetition index, and the activity boundaries `pb` and `pe` taken from the JSON `groundTruth` field. Training and testing repetition blocks are merged into a single repetition axis per user.

### Step 2: Windowing

```bash
python process_epn612.py
```

Runs once. Windows are `(N, 8, 40)`: 8 channels, 40 samples at 200 Hz (200 ms), stride `INC = 5` samples (25 ms). Rest windows are subsampled 4:1 after windowing in all four variants. Four variants are produced:

| Variant | Construction |
|---|---|
| `raw` | Windows over the complete repetition, no temporal curation. Default for training and for the primary test set. |
| `segmented` | Energy-based active-region detector per active repetition, using the squared amplitude of the highest-energy channel, smoothed, with a relative threshold of 0.25. Frames outside the detected region are discarded. Rest repetitions are untouched. |
| `relabeled` | Same boundaries as `segmented`, but the pre-onset and post-offset frames are relabelled as rest instead of discarded, which widens the rest distribution toward realistic transition data. |
| `standard` | Population-level amplitude threshold (training rest mean plus 3 standard deviations) applied through a sliding-window energy detector. The threshold comes from training data only and is applied identically to all splits. |

Each variant writes to `pickles/`:

```
{train,val,test}_windows_{variant}.npy    (N, 8, 40) float32
{train,val,test}_meta_{variant}.npy       dict: classes, subjects, reps, base_class
{train,val,test}_data_{variant}.npy       pickled OfflineDataHandler, used by within-user scripts
```

## Models

Defined in `models.py`. Raw-input models apply `x / 128.0` internally.

**MHCNN** (proposed, about 89k parameters). Three parallel `Conv1d` branches over the raw 8-channel window, kernel size 8, dilations 1, 2, and 4, giving receptive fields of roughly 40, 80, and 160 ms in parallel rather than in sequence. Branch outputs are concatenated to 96 channels, passed through a fourth `Conv1d` with kernel size 4, then `AdaptiveAvgPool1d(1)`, a 128-unit GELU layer, and a 128-dimensional embedding. The classifier is a single linear layer on the embedding. Adaptive pooling over time is what makes the representation insensitive to where in the window the contraction sits.

**MHCNN_GRL**. MHCNN plus a gradient reversal layer and a 306-way user-identity head, for domain-adversarial training against user identity.

**CNN**. Single-scale sequential `Conv1d` stack (kernels 4, 3, 3). Ablation that isolates the contribution of parallel multi-horizon dilation.

**LSTM**. Three-layer bidirectional LSTM over the 40 raw timesteps.

**MLP**. Three-layer fully connected network on full-window hand-crafted features.

**LSTM_HCF**. Three-layer LSTM on sub-windowed features. The 40-sample window is split into `N_SUB = 4` sub-windows of 10 samples, giving a `(4, F)` sequence.

**CNN_HCF**. 1D CNN on the same sub-windowed features, transposed to `(B, F, 4)` so features act as channels and sub-windows as time.

## Loss functions

All losses take `(logits, labels, ...)` or `(emb, logits, labels)` and are interchangeable through the `CONFIGS` dictionary in the training scripts.

| Key | Class | Description |
|---|---|---|
| `base` | `BaseLoss` | Cross entropy |
| `rest` | `RestLoss` | CE reweighted per sample for wrong-active predictions and for rest predicted as active |
| `act` | `ActiveLoss` | CE reweighted toward active-gesture samples |
| `cvar` | `CVaRLoss` | CE restricted to the hardest `alpha` fraction of each batch (default 0.3) |
| `std` | `STDLoss` | Class-balanced CE mean plus a penalty on the batch loss standard deviation |
| `sbj` | `PerSubjectLoss` | Per-user mean loss plus per-user loss standard deviation, computed within batch |
| `proto` | `PrototypeLoss` | CE plus within-class prototype compactness, no repulsion |
| `1va` | `OneVsAllLoss` | CE plus prototypical CE over negative squared distances to class means, compactness and repulsion |
| `ang` | `AngularLoss` | CE plus supervised contrastive term plus an angular penalty between active-class directions measured from the rest prototype |
| `trp` | `TripletLoss` | CE plus batch-hard triplet loss, with a sampler that guarantees cross-user positives and same-user negatives |
| `grl` | `MHCNN_GRL` + CE | Domain-adversarial user-identity confusion through gradient reversal |

`AngularLoss` exists because the embedding geometry is radial, not spherical: active gestures fan outward from the rest cluster along distinct directions, and contraction intensity moves a window along its ray without changing its class. Losses that assume isotropic clusters penalise legitimate intensity spread.

## Augmentation

`AugConfig` and `EMGAugment` in `utils.py` implement on-device augmentation aimed at the physical sources of cross-user variability that a larger training population would otherwise have to supply:

- **rotate**: cyclic channel shift, modelling a different armband donning rotation around the forearm.
- **gain**: log-uniform per-channel and per-sample amplitude scaling, modelling electrode impedance, tissue thickness, and effort. This targets the user-specific amplitude cue the encoder otherwise latches onto.
- **warp**: smooth multiplicative envelope from random knots, modelling non-stationary effort within a 200 ms window.
- **noise**: zero-mean Gaussian at a per-channel SNR plus an absolute floor.

All transforms operate on the raw ADC scale, are applied per sample through boolean masks and `torch.where` with no host-device synchronisation, and are clamped back into the signed 8-bit range. `train_aug` builds the augmented batch with `n_aug` copies and an optional clean copy retained.

## Training scripts

Every script takes the visible GPU index as its first argument.

### `cross_mhcnn.py`

Proposed model plus the full loss benchmark, evaluated against all four test variants.

```bash
python cross_mhcnn.py <GPU> <TAG> <norm|nonorm> <variant[,variant,...]|all> <train|eval>
```

```bash
# all loss variants, raw windows
python cross_mhcnn.py 0 raw nonorm all train

# two variants only
python cross_mhcnn.py 0 raw nonorm base,trp train

# reload checkpoints and evaluate under RunningNorm
python cross_mhcnn.py 0 raw norm base eval
```

Run names follow `cross_mhcnn_{TAG}_{variant}` with a `-rn` suffix when `RunningNorm` is active.

`RunningNorm` is a streaming per-channel normaliser initialised from population statistics computed on training windows, updated by exponential moving average at inference. `TAU = inf` gives an exact cumulative mean; a finite `tau` gives a fixed-length EMA. It adapts to a new user's channel statistics without labels, which keeps the setup calibration-free, and it is the only per-user adaptation in the codebase. `RN_PRIOR_WEIGHT` sets how much the population prior resists the first batches.

### `cross_feats.py`

Feature-set grid search under DDP: LDA, MLP, LSTM_HCF, and CNN_HCF over 11 feature groups (WENG, RMS, HTD, DFTR, ITD, LS4, TDAR, COMB, MSWT, TDPSD, LS9).

```bash
torchrun --nproc_per_node=<N> cross_feats.py <GPU> <TAG> <FEAT[,FEAT,...]>
```

WENG (wavelet energy, one feature per channel) was selected from this search and is hard-coded in the downstream feature-based scripts.

### `cross_models.py`

Architecture comparison on the fixed split under DDP: LDA, MLP, LSTM_HCF, CNN_HCF, LSTM, CNN, MHCNN, all on WENG features or raw windows as appropriate.

```bash
torchrun --nproc_per_node=<N> cross_models.py <GPU>
```

### `inc_mhcnn.py` and `inc_mhcnn_aug.py`

Data-scaling sweep over training-user count (1, 2, 4, 8, 16, 32, 64, 128, 196, 306) and repetitions per gesture (1, 2, 4, 8, 16, 24, 32, 40, 50), for three user-ordering strategies.

```bash
python inc_mhcnn.py <GPU> <TAG> <run[,run,...]> <loss_variant>
python inc_mhcnn_aug.py <GPU> <TAG> <run[,run,...]> <loss_variant>
```

Run 0 is worst-user-first ordering, run 1 is best-first, runs above 2 are deterministic random permutations. Orderings 0 and 1 rank users by within-user balanced accuracy taken from `checkpoints/within_mhcnn_raw_base/results-15.npy`, so `within_mhcnn.py` must have been run first. Batch size is scaled down for small subsets. Per-cell results append to `figures/{name}-{run}.csv`.

`inc_mhcnn_aug.py` is the same sweep with `EMGAugment` in the loop (`N_AUG = 2`, `KEEP_CLEAN = True`), which isolates how much synthetic variability substitutes for real users.

### Within-user scripts

Within-user models set the achievable ceiling for each user and provide the difficulty ranking used elsewhere.

```bash
python within_mhcnn.py <GPU> <TAG> <rep[,rep,...]> <ft|noft>
python within_cnnhcf.py <GPU> <TAG> <rep[,rep,...]> <ft|noft>
python within_lstmhcf.py <GPU> <TAG> <rep[,rep,...]> <ft|noft>
python within_mlp.py <GPU> <TAG> <rep[,rep,...]> <ft|noft>
python within_lda.py <rep[,rep,...]>
```

One model per user across all 612 users. The repetition argument sets how many repetitions per gesture are used for training; held-out repetitions provide validation and test. `ft` initialises from a cross-user checkpoint instead of from scratch, which measures how much a calibration-free model is worth as a starting point. These scripts override the global hyperparameters with `BATCH_SIZE = 64`, `PATIENCE = 5`, `LR_PATIENCE = 3`.

## Evaluation

`eval_test` computes four metrics per test user and then aggregates across users:

- accuracy over all windows
- active accuracy, restricted to non-rest windows
- balanced accuracy, macro-averaged per-class recall
- macro F1

Per-user values are stacked as a `(4, n_subjects)` array in that order, so index 2 is balanced accuracy. Outputs:

```
checkpoints/{name}/results_{tag}.npy    (4, n_subjects) per-user metrics
checkpoints/{name}/preds_{tag}.npy      window-level predictions
checkpoints/{name}/labels_{tag}.npy     window-level labels
checkpoints/{name}/{name}.pt            best-epoch weights
figures/{name}/{tag}.jpg                per-user bar charts, sorted by balanced accuracy
figures/results.csv                     appended aggregate row per run and test set
```

A model trained on one variant is evaluated on all four test variants, which quantifies how much of the reported performance depends on test-side curation rather than on the model.

Statistical comparison between models follows the same protocol used in the manuscripts: Friedman omnibus over per-subject scores, all-pairs Wilcoxon signed-rank post-hoc, Holm-Bonferroni correction, Cohen's d effect sizes, always paired across the 280 test users.

## Analysis

### `Analysis_PCA.py`

```bash
python Analysis_PCA.py <GPU> <TAG> <LOSS>
```

Loads per-epoch checkpoints from a `cross_mhcnn.py` run, fits PCA on the best-epoch test embeddings, and writes projections to `figures/{name}_PCAs_{dims}/`. Used to check whether hard users occupy a separable region of the embedding space or are mixed through it.

### `Analysis_Taxonomy.py`

```bash
python Analysis_Taxonomy.py <GPU> <SPACE[,SPACE]> <RESULTS_TAG>
```

Per-user failure-mode analysis in hand-crafted feature space, run independently per feature space (WENG and HTD by default). For each test user, features are projected onto four principal components, both a within-user basis and a population basis, and polar coordinates are taken around the coordinate-wise median of that user's rest class. Two indices are computed:

- **ASI**, angular separability index: cosine silhouette over the four active classes on unit directions from the rest origin. Rest is excluded, so this measures gesture identity only. Cosine rather than Euclidean, because identity is the angular coordinate of the fan and Euclidean distance would charge contraction-intensity spread as a class-compactness failure. Low ASI indicates Type 1 failure, intrinsic signal inseparability.
- **RLI**, rest leakage index: the fraction of active-labelled windows whose radius falls inside the rest envelope, defined by the 90th percentile of rest radii. The direction matters: this counts active-labelled windows intruding on rest, not rest extending into active regions. High RLI indicates Type 2 failure, behavioural or segmentation leakage, which is the removable part.

Thresholds are derived from the reference cohort (top quartile of cross-user balanced accuracy): ASI at the 25th percentile of reference ASI, RLI at the 75th percentile of reference RLI. The hard cohort is the bottom quartile, about 70 users. Thresholds are recomputed independently for each feature space.

The script writes `figures/taxonomy/taxonomy_indices.csv`, `taxonomy_report.txt`, a quadrant figure per feature space, and index-versus-accuracy scatter plots, and it reports the raw to segmented RLI shift with a Wilcoxon test to confirm that segmentation removes the leakage population it is supposed to remove.

### `eval_sgt_cross.py`

```bash
python eval_sgt_cross.py <GPU>
```

Applies EPN-trained cross-user checkpoints to the screen-guided training data collected from the online Fitts' law study participants, held under `user_sgt/{user_id}/`, using the held-out validation repetition. Handles the class-order remapping between the online collection protocol and the EPN label order, optionally applies the same energy segmentation used for the `segmented` variant, and optionally wraps the model in `RunningNorm` with the population statistics hard-coded at the top of the file. This is the offline bridge between the EPN-612 benchmark and the online study cohort.

## Hyperparameters

Defined in `utils.py`. The within-user scripts override several of these locally, as noted above.

| Parameter | Value | Meaning |
|---|---|---|
| `SEQ` | 40 | Window length in samples (200 ms at 200 Hz) |
| `INC` | 5 | Window stride in samples (25 ms) |
| `CH` | 8 | Channels |
| `CLASSES` | 5 | Gesture classes |
| `N_SUB` | 4 | Sub-windows for feature-sequence models |
| `VAL_CUTOFF` | 332 | 0-indexed validation and test boundary |
| `BATCH_SIZE` | 2048 | Training batch size |
| `EPOCHS` | 300 | Maximum epochs |
| `PATIENCE` | 15 | Early-stopping patience on validation loss |
| `LR_INIT` | 5e-4 | Adam initial learning rate |
| `LR_FACTOR` | 0.6 | ReduceLROnPlateau factor |
| `LR_PATIENCE` | 7 | ReduceLROnPlateau patience |
| `LR_MIN` | 1e-6 | Learning-rate floor |
| `DROPOUT` | 0.2 | Dropout rate |
| `MARGIN` | 0.5 | Triplet margin |
| `W_HARD`, `W_SOFT` | 1.0, 0.0 | Batch-hard and soft-margin triplet weights |
| `ALPHA_START`, `ALPHA_END`, `WARMUP` | 0.01, 0.25, 25 | Triplet term ramp |
| `TAU` | inf | RunningNorm time constant (inf gives cumulative mean) |
| `RN_PRIOR_WEIGHT` | 1 | Prior weight on the population statistics in RunningNorm |
| `SEED` | 13 | Base seed in every script |

Training uses Adam, AMP autocast with `GradScaler`, gradient clipping, `ReduceLROnPlateau` on validation loss, and early stopping with restoration of the best-epoch weights.

## What the benchmark has established so far

- Multi-horizon parallel dilation with adaptive temporal pooling on raw EMG outperforms the hand-crafted feature pipelines and the single-scale and recurrent baselines on the cross-user split.
- WENG matched or exceeded every multi-feature set tested, within a small margin, so feature-set choice is not the bottleneck.
- Every loss modification listed above is statistically indistinguishable from plain cross entropy after correction for multiple comparisons. The encoder learns user-specific amplitude structure from the input before any loss term acts, so loss design alone does not move the cross-user floor.
- Hard users are not a single population. Type 1 users, low ASI, are intrinsically inseparable in feature space and are invariant to loss choice; their windows are mixed through the embedding space rather than clustered apart. Type 2 users, high RLI, carry removable label leakage and respond to active-region segmentation.
- Ensembling many independently seeded models yields marginal gains and does not close the tail, which supports the Type 1 interpretation.

Numeric results, statistics, and figures for these points live in the thesis and manuscripts, not in this repository.

## Author

Amir Hariri, Institute of Biomedical Engineering, University of New Brunswick.

## License

MIT. See [LICENSE](LICENSE).