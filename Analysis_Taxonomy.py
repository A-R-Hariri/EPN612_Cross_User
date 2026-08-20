import warnings, sys, os, gc
from os.path import join
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"

import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats
from itertools import combinations
import matplotlib.pyplot as plt

from utils import *


# ======== CONFIG ========
SEED = 13
RNG = np.random.default_rng(SEED)
MMAP_MODE = 'r'

TAGS = ['raw'] 
SPACES = sys.argv[2].split(',') if len(sys.argv) > 2 else ['WENG', 'HTD']
RESULTS_TAG = sys.argv[3] if len(sys.argv) > 3 else 'cross_mhcnn_raw_base'

N_PC = 4
POP_FIT_ROWS = 400000
MAX_SIL_WINDOWS = 6000
MIN_CLASS_WINDOWS = 20

VAL_CUTOFF = 332
EXEMPLARS_REL = [22, 235, 74, 75]
HARD_QUARTILE = 25
EASY_QUARTILE = 75

OUT_DIR = join(FIGURE_PATH, 'taxonomy')
os.makedirs(OUT_DIR, exist_ok=True)

ACTIVE = np.array([1, 2, 3, 4])
CLASS_NAMES = ['NM', 'HC', 'FX', 'EX', 'HO']


# ======== FEATURE LOADING ========
def feat_path(split, space, tag):
    return join(PICKLE_PATH, f'{split}_{space.lower()}_{tag}.npy')


def load_feats(split, space, tag):
    F = np.load(feat_path(split, space, tag), mmap_mode=MMAP_MODE)
    return F.reshape(F.shape[0], -1)


def load_meta(split, tag):
    return np.load(join(PICKLE_PATH, f'{split}_meta_{tag}.npy'), allow_pickle=True).item()


# ======== POPULATION BASIS ========

def fit_population_basis(space, tag):
    F = load_feats('train', space, tag)
    n = F.shape[0]
    if n > POP_FIT_ROWS:
        idx = np.sort(RNG.choice(n, POP_FIT_ROWS, replace=False))
        Z = np.asarray(F[idx])
    else:
        Z = np.asarray(F)
    pca_pop = PCA(n_components=N_PC, random_state=SEED).fit(Z)
    del F, Z; gc.collect()
    return pca_pop


# ======== GEOMETRY ========

def polar_coords(P, y):
    # Origin is the coordinate-wise median of the rest class, not the mean.
    # Rest itself is not moved by curation (segmentation never touches rest
    # repetitions), so the origin is stable across raw/segmented; what changes
    # is which active windows fall near it.
    c0 = np.median(P[y == 0], axis=0)
    V = P - c0
    r = np.linalg.norm(V, axis=1)
    U = V / np.clip(r, 1e-8, None)[:, None]
    return r, U


def subsample(y, keep_mask, n_max):
    idx = np.flatnonzero(keep_mask)
    if idx.size <= n_max:
        return idx
    per = max(1, n_max // np.unique(y[idx]).size)
    out = []
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c]
        out.append(RNG.choice(ci, min(per, ci.size), replace=False))
    return np.sort(np.concatenate(out))


# ======== METRICS ========

def angular_separability(U, y):
    # ASI. Cosine silhouette over the four active classes on unit directions
    # from the rest origin. Rest excluded, so this is gesture identity only.
    # Cosine and not Euclidean because identity is the angular coordinate of the
    # fan; Euclidean would charge legitimate contraction-
    # intensity spread as a class-compactness failure.
    m = np.isin(y, ACTIVE)
    idx = subsample(y, m, MAX_SIL_WINDOWS)
    ya = y[idx]
    if np.unique(ya).size < 2:
        return np.nan
    counts = np.array([np.sum(ya == c) for c in np.unique(ya)])
    if counts.min() < MIN_CLASS_WINDOWS:
        return np.nan
    return float(silhouette_score(U[idx], ya, metric='cosine'))


def auc_1d(t, g):
    a, b = t[g], t[~g]
    if a.size < MIN_CLASS_WINDOWS or b.size < MIN_CLASS_WINDOWS:
        return np.nan
    u = stats.mannwhitneyu(a, b, alternative='two-sided').statistic
    return float(u / (a.size * b.size))


def rest_leakage(r, y, rest_pctile=90):
    # RLI. Fraction of ACTIVE-labeled windows whose radius falls inside the
    # rest cluster's envelope (radius <= the rest_pctile percentile of rest
    # radii). Segmentation only crops active repetitions and leaves rest
    # repetitions untouched, so it cannot move or tighten the rest cluster
    # itself; what it removes is exactly the low-amplitude, active-labeled
    # antagonist-rebound windows that sit inside that zone. RLI is defined to
    # track that removable population directly, not the (unmovable) rest
    # cluster shape. Bounded in [0, 1]; 0 means no active window intrudes on
    # the rest envelope.
    g = np.isin(y, ACTIVE)
    if (~g).sum() < MIN_CLASS_WINDOWS or g.sum() < MIN_CLASS_WINDOWS:
        return np.nan, np.nan
    rest_thr = np.percentile(r[~g], rest_pctile)
    frac = float(np.mean(r[g] <= rest_thr))
    a = auc_1d(r, g)
    auc_leak = float(1.0 - max(a, 1.0 - a)) if not np.isnan(a) else np.nan
    return frac, auc_leak


def pairwise_active_auc(U, y):
    # Per active pair, project directions onto the line joining the two class
    # direction-means and take the AUC. Resolves whether entanglement is global
    # or confined to specific pairs.
    out = {}
    for a, b in combinations(ACTIVE, 2):
        ma, mb = y == a, y == b
        if ma.sum() < MIN_CLASS_WINDOWS or mb.sum() < MIN_CLASS_WINDOWS:
            out[f'auc_{CLASS_NAMES[a]}_{CLASS_NAMES[b]}'] = np.nan
            continue
        w = U[ma].mean(0) - U[mb].mean(0)
        nw = np.linalg.norm(w)
        if nw < 1e-8:
            out[f'auc_{CLASS_NAMES[a]}_{CLASS_NAMES[b]}'] = 0.5
            continue
        t = U[ma | mb] @ (w / nw)
        g = y[ma | mb] == a
        v = auc_1d(t, g)
        out[f'auc_{CLASS_NAMES[a]}_{CLASS_NAMES[b]}'] = np.nan if np.isnan(v) else max(v, 1.0 - v)
    vals = np.array([v for v in out.values() if not np.isnan(v)])
    out['auc_pair_mean'] = float(vals.mean()) if vals.size else np.nan
    out['auc_pair_min'] = float(vals.min()) if vals.size else np.nan
    return out


def fisher_ratio(U, y):
    # Trace ratio of between- to within-class scatter over the actives, on the
    # directions.
    m = np.isin(y, ACTIVE)
    Ua, ya = U[m], y[m]
    if np.unique(ya).size < 2:
        return np.nan
    mu = Ua.mean(0)
    sb = sw = 0.0
    for c in np.unique(ya):
        Uc = Ua[ya == c]
        d = Uc.mean(0) - mu
        sb += Uc.shape[0] * float(d @ d)
        sw += float(((Uc - Uc.mean(0)) ** 2).sum())
    return float(sb / sw) if sw > 1e-12 else np.nan


# ======== PER-USER PIPELINE ========

def user_row(Zu, y, pca_pop, subject_abs, tag, space):
    row = {'subject': int(subject_abs),
           'subject_rel': int(subject_abs - VAL_CUTOFF),
           'tag': tag, 'space': space, 'n_windows': int(Zu.shape[0])}

    pca_u = PCA(n_components=N_PC, random_state=SEED).fit(Zu)
    projections = {
        'within': (pca_u.transform(Zu), pca_u.explained_variance_ratio_),
        'cross': (pca_pop.transform(Zu), pca_pop.explained_variance_ratio_),
    }

    for fit, (P, evr) in projections.items():
        r, U = polar_coords(P, y)
        rli, leak = rest_leakage(r, y)
        row[f'asi_{fit}'] = angular_separability(U, y)
        row[f'rli_{fit}'] = rli
        row[f'leakfrac_{fit}'] = leak
        row[f'fisher_{fit}'] = fisher_ratio(U, y)
        row[f'evr4_{fit}'] = float(evr[:N_PC].sum())
        for k in range(N_PC):
            row[f'evr_pc{k + 1}_{fit}'] = float(evr[k])
        if fit == 'within':
            for k, v in pairwise_active_auc(U, y).items():
                row[f'{k}_{fit}'] = v

    # Sensitivity: same two indices in the full feature space with no PCA, to
    # show the taxonomy is not an artifact of truncating to four components.
    r, U = polar_coords(Zu, y)
    rli, _ = rest_leakage(r, y)
    row['asi_fullD'] = angular_separability(U, y)
    row['rli_fullD'] = rli
    return row


def run_tag_space(tag, space):
    pca_pop = fit_population_basis(space, tag)
    F = load_feats('test', space, tag)
    meta = load_meta('test', tag)
    subjects = np.asarray(meta['subjects'])
    classes = np.asarray(meta['classes'])

    rows = []
    for s in np.unique(subjects):
        m = subjects == s
        Zu = np.asarray(F[m])
        rows.append(user_row(Zu, classes[m], pca_pop, s, tag, space))
        del Zu; gc.collect()
        print(f'{tag} | {space} | subject {s} (rel {int(s) - VAL_CUTOFF}) done')

    del F; gc.collect()
    return pd.DataFrame(rows)


# ======== POPULATION ANALYSIS ========

def load_bal_acc(tag):
    p = join(CHECKPOINT_PATH, RESULTS_TAG, f'results_{tag}.npy')
    if not os.path.exists(p):
        p = join(CHECKPOINT_PATH, RESULTS_TAG, 'results.npy')
    res = np.asarray(np.load(p, allow_pickle=True))
    return res[2]


def bal_for(df_group, bal):
    return bal[df_group.subject_rel.to_numpy()]


def spearman(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    rho, p = stats.spearmanr(a[m], b[m])
    return float(rho), float(p), int(m.sum())


def partial_spearman(a, b, c):
    m = ~(np.isnan(a) | np.isnan(b) | np.isnan(c))
    ra, rb, rc = (stats.rankdata(v[m]) for v in (a, b, c))
    resid = lambda v: v - np.polyval(np.polyfit(rc, v, 1), rc)
    rho, p = stats.pearsonr(resid(ra), resid(rb))
    return float(rho), float(p), int(m.sum())


def rank_biserial(a, b):
    u = stats.mannwhitneyu(a, b, alternative='two-sided')
    return float(1.0 - 2.0 * u.statistic / (a.size * b.size)), float(u.pvalue)


def population_report(df, bal):
    d = df[df.tag == 'raw'].copy()
    lines = []
    for space in SPACES:
        g = d[d.space == space].sort_values('subject_rel')
        g = g.assign(bal_acc=bal_for(g, bal))

        lines.append(f'==== {space} ====')
        for fit in ['within', 'cross']:
            for idx in ['asi', 'rli']:
                rho, p, n = spearman(g[f'{idx}_{fit}'].to_numpy(), g.bal_acc.to_numpy())
                lines.append(f'{idx}_{fit} vs bal_acc: rho={rho:.3f} p={p:.3g} n={n}')

        rho, p, n = spearman(g.asi_within.to_numpy(), g.rli_within.to_numpy())
        lines.append(f'asi_within vs rli_within: rho={rho:.3f} p={p:.3g} n={n}')

        rho, p, n = partial_spearman(g.asi_within.to_numpy(), g.bal_acc.to_numpy(),
                                     g.rli_within.to_numpy())
        lines.append(f'asi | rli partialled, vs bal_acc: rho={rho:.3f} p={p:.3g} n={n}')
        rho, p, n = partial_spearman(g.rli_within.to_numpy(), g.bal_acc.to_numpy(),
                                     g.asi_within.to_numpy())
        lines.append(f'rli | asi partialled, vs bal_acc: rho={rho:.3f} p={p:.3g} n={n}')

        lo = np.percentile(g.bal_acc, HARD_QUARTILE)
        hard = g[g.bal_acc <= lo]
        easy = g[g.bal_acc > lo]
        for idx in ['asi_within', 'rli_within']:
            rb, p = rank_biserial(hard[idx].dropna().to_numpy(), easy[idx].dropna().to_numpy())
            lines.append(f'{idx} hard vs rest: rank-biserial={rb:.3f} p={p:.3g} '
                         f'hard_med={hard[idx].median():.3f} easy_med={easy[idx].median():.3f}')

        # Mode assignment. Thresholds set from the top-quartile reference users, so
        # a hard user is labelled Type 1 or Type 2 only when their geometry falls
        # outside the range the easy users occupy.
        ref = g[g.bal_acc >= np.percentile(g.bal_acc, EASY_QUARTILE)]
        asi_thr = float(np.nanpercentile(ref.asi_within, HARD_QUARTILE))
        rli_thr = float(np.nanpercentile(ref.rli_within, EASY_QUARTILE))
        t1 = hard.asi_within < asi_thr
        t2 = hard.rli_within > rli_thr
        lines.append(f'thresholds from top quartile: asi<{asi_thr:.3f} rli>{rli_thr:.3f}')
        lines.append(f'hard users n={len(hard)}: '
                     f'type1_only={int((t1 & ~t2).sum())} '
                     f'type2_only={int((~t1 & t2).sum())} '
                     f'mixed={int((t1 & t2).sum())} '
                     f'neither={int((~t1 & ~t2).sum())}')
        lines.append(f'EVR of 4 PCs: within={g.evr4_within.mean():.3f} '
                     f'cross={g.evr4_cross.mean():.3f}')
        lines.append('')

        t1 = hard.asi_within < asi_thr
        t2 = hard.rli_within > rli_thr
        print(f'{space} Type2 users:', hard.loc[(~t1 & t2), 'subject_rel'].tolist())

        quadrant_figure(g, asi_thr, rli_thr, space)
    return '\n'.join(lines)


def exemplar_table(df, bal):
    d = df[df.tag == 'raw']
    cols = ['subject_rel', 'subject', 'space', 'asi_within', 'asi_cross',
            'rli_within', 'rli_cross', 'fisher_within', 'auc_pair_min_within',
            'auc_pair_mean_within', 'evr4_within']
    t = d[d.subject_rel.isin(EXEMPLARS_REL)][cols].copy()
    t['bal_acc'] = bal[t.subject_rel.to_numpy()]  # already a percentage
    t['order'] = t.subject_rel.map({s: i for i, s in enumerate(EXEMPLARS_REL)})
    return t.sort_values(['order', 'space']).drop(columns='order')


def compute_segmented_rli():
    cache = join(OUT_DIR, 'segmented_rli.csv')
    try:
        return pd.read_csv(cache)
    except:
        pass
    rows = []
    for space in SPACES:
        pca_pop = fit_population_basis(space, 'segmented')
        F = load_feats('test', space, 'segmented')
        meta = load_meta('test', 'segmented')
        subjects = np.asarray(meta['subjects'])
        classes = np.asarray(meta['classes'])
        for s in np.unique(subjects):
            m = subjects == s
            Zu = np.asarray(F[m])
            y = classes[m]
            pca_u = PCA(n_components=N_PC, random_state=SEED).fit(Zu)
            r, _ = polar_coords(pca_u.transform(Zu), y)
            rli, _ = rest_leakage(r, y)
            rows.append({'subject_rel': int(s - VAL_CUTOFF), 'space': space,
                         'rli_within': rli})
            del Zu; gc.collect()
            print(f'segmented | {space} | subject {s} (rel {int(s) - VAL_CUTOFF}) RLI done')
        del F; gc.collect()
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out


def rli_curation_report(df_raw, df_seg):
    lines = []
    for space in SPACES:
        base = df_raw[(df_raw.tag == 'raw') & (df_raw.space == space)].set_index('subject_rel')
        other = df_seg[df_seg.space == space].set_index('subject_rel')
        common = base.index.intersection(other.index)
        a = base.loc[common, 'rli_within'].to_numpy()
        b = other.loc[common, 'rli_within'].to_numpy()
        m = ~(np.isnan(a) | np.isnan(b))
        w = stats.wilcoxon(a[m], b[m])
        dd = b[m] - a[m]
        dz = float(dd.mean() / dd.std(ddof=1)) if dd.std(ddof=1) > 1e-12 else np.nan
        lines.append(f'{space} | rli_within raw->segmented: '
                     f'{a[m].mean():.4f} -> {b[m].mean():.4f} '
                     f'delta={dd.mean():+.4f} W={w.statistic:.0f} '
                     f'p={w.pvalue:.3g} dz={dz:.2f} n={m.sum()}')
    return '\n'.join(lines)


# ======== FIGURES ========

def quadrant_figure(g, asi_thr, rli_thr, space):
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=200)
    sc = ax.scatter(g.asi_within, g.rli_within, c=g.bal_acc, cmap='viridis',
                    s=18, linewidths=0, alpha=0.85)
    ax.axvline(asi_thr, color='0.4', lw=0.8, ls='--')
    ax.axhline(rli_thr, color='0.4', lw=0.8, ls='--')
    for s in EXEMPLARS_REL:
        p = g[g.subject_rel == s]
        if len(p):
            ax.scatter(p.asi_within, p.rli_within, facecolors='none',
                       edgecolors='crimson', s=90, linewidths=1.2)
            ax.annotate(f'U{s}', (float(p.asi_within.iloc[0]), float(p.rli_within.iloc[0])),
                        textcoords='offset points', xytext=(6, 5), fontsize=8, color='crimson')
    ax.set_xlabel('Angular separability index (Type 1 axis)')
    ax.set_ylabel('Rest leakage index (Type 2 axis)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.colorbar(sc, ax=ax, label='Cross-user balanced accuracy (%)')
    fig.tight_layout()
    fig.savefig(join(OUT_DIR, f'taxonomy_quadrant_{space.lower()}.png'), bbox_inches='tight')
    plt.close(fig)


def index_vs_accuracy_figure(g, space):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), dpi=200)
    for ax, idx, lab in zip(axes, ['asi_within', 'rli_within'],
                            ['Angular separability index', 'Rest leakage index']):
        ax.scatter(g[idx], g.bal_acc, s=14, linewidths=0, alpha=0.7, color='#4C72B0')
        rho, p, n = spearman(g[idx].to_numpy(), g.bal_acc.to_numpy())
        ax.set_title(f'rho = {rho:.2f}', fontsize=9)
        ax.set_xlabel(lab)
        ax.set_ylabel('Balanced accuracy (%)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(join(OUT_DIR, f'taxonomy_indices_vs_acc_{space.lower()}.png'), bbox_inches='tight')
    plt.close(fig)


# ======== MAIN ========

if __name__ == '__main__':
    try:
        df = pd.read_csv(join(OUT_DIR, 'taxonomy_indices.csv'))
    except:
        frames = []
        for tag in TAGS:
            for space in SPACES:
                frames.append(run_tag_space(tag, space))
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(join(OUT_DIR, 'taxonomy_indices.csv'), index=False)

    bal = load_bal_acc('raw')

    report = []
    report.append('======== POPULATION ========')
    report.append(population_report(df, bal))
    report.append('======== EXEMPLARS ========')
    report.append(exemplar_table(df, bal).to_string(index=False, float_format='%.3f'))
    report.append('======== RLI: RAW vs SEGMENTED ========')
    report.append(rli_curation_report(df, compute_segmented_rli()))
    text = '\n'.join(report)

    with open(join(OUT_DIR, 'taxonomy_report.txt'), 'w') as f:
        f.write(text)
    print(text)

    for space in SPACES:
        g = df[(df.tag == 'raw') & (df.space == space)].sort_values('subject_rel')
        g = g.assign(bal_acc=bal[g.subject_rel.to_numpy()])
        index_vs_accuracy_figure(g, space)