# Cross-User Zero-Shot EMG Gesture Classification on EPN612

End-to-end model design and benchmark for cross-user EMG gesture classification. Models train on a population of users and are evaluated zero-shot on held-out subjects with no per-user calibration. This is the precondition for any EMG interface that ships beyond the lab, since per-user calibration breaks under electrode shift, sweat, fatigue, and time.

## Dataset

EPN-612: 612 subjects, 5 hand gestures plus rest, 8-channel Myo armband at 200 Hz.

## Methods

- Architectures: LDA, MLP, CNN, LSTM, Transformer
- Input representations: raw windows, segmented windows, hand-crafted feature sets
- Cross-user training with embedding-based variants (metric learning, gradient reversal)
- Within-user baselines and fine-tuning ablations (standard, segmented, raw)
- Incremental reps/users experiments

## Results

Best cross-user zero-shot accuracy from 77% baseline to **81%** on held-out subjects with no calibration. Within-user upper bound **90%**.

## Repo layout

- `process_epn612.py`, `EPN612.py`: dataset ingestion and windowing
- `models.py`, `utils.py`: shared model, losses, and training utilities
- `cross_feats.py`: feature grid search
- `cross_models.py`: cross-user model comparison
- `cross_mhcnn.py`: cross-user training for the proposed model and losses
- `within_*.py`: within-user baselines and fine-tuning ablations
- `inc_mhcnn.py`: incremental reps/subjects trauining for analysis
- `Analysis_PCA.py`: PCA projection of the latent space of trained model from each epoch's checkpoint on the test set

## Author

Amir Hariri, Institute of Biomedical Engineering, University of New Brunswick.
