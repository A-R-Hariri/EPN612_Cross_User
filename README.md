# Cross-User Zero-Shot EMG Gesture Classification on EPN612

End-to-end benchmark for user-invariant EMG gesture classification. Models train on a population of users and are evaluated zero-shot on held-out subjects with no per-user calibration. This is the precondition for any EMG interface that ships beyond the lab, since per-user calibration breaks under electrode shift, sweat, fatigue, and time.

## Dataset

EPN-612: 612 subjects, 5 hand gestures plus rest, 8-channel Myo armband at 200 Hz.

## Methods

- Architectures: LDA, MLP, CNN, LSTM, Transformer
- Input representations: raw windows, segmented windows, hand-crafted feature sets
- Cross-user training with embedding-based variants (metric learning, gradient reversal)
- Within-user baselines and fine-tuning ablations (standard, segmented, raw)
- Incremental user adaptation experiments

## Results

Best cross-user zero-shot accuracy **80%** on held-out subjects with no calibration. Within-user upper bound **85%**.

## Repo layout

- `process_epn612.py`, `EPN612.py`: dataset ingestion and windowing
- `models.py`, `utils.py`: shared model and training utilities
- `cross_*.py`: cross-user training per architecture
- `within_*.py`: within-user baselines and fine-tuning ablations
- `inc_*_.py`: incremental reps/subjects trauining for analysis

## Author

Amir Hariri, Institute of Biomedical Engineering, University of New Brunswick.
