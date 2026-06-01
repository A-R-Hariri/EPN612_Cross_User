import torch
import torch.nn as nn; 
import torch.nn.functional as F
from torch.autograd import Function

from utils import *

    
# ======== MODELS ========

# -------- Proposed --------
class MHCNN(nn.Module):
    """
    Proposed model: parallel dilated multi-horizon CNN on raw EMG.
    Three parallel branches (dilation 1, 2, 4) capture activation
    dynamics at ~40ms, ~80ms, and ~160ms receptive fields simultaneously,
    covering the full temporal structure available within a 200ms window.

    Input : (B, 8, 40) 
    """
    def __init__(self, ch: int = CH, emb_dim: int = 128,
                 num_classes: int = CLASSES, dropout: float = DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, 32, 8, dilation=1, padding='same')
        self.conv2 = nn.Conv1d(ch, 32, 8, dilation=2, padding='same')
        self.conv3 = nn.Conv1d(ch, 32, 8, dilation=4, padding='same')
        self.conv4 = nn.Conv1d(96, 128, 4, dilation=1, padding='same')
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc1        = nn.Linear(128, 128)
        self.fc_emb     = nn.Linear(128, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, return_emb=False, return_logits=False):
        # x: (B, 8, 40)
        x = x / 128.0       # Scaling, not normalization
        x1 = self.drop(self.relu(self.conv1(x)))
        x2 = self.drop(self.relu(self.conv2(x)))
        x3 = self.drop(self.relu(self.conv3(x)))
        x  = self.drop(self.relu(self.conv4(torch.cat((x1, x2, x3), dim=1))))
        x  = self.pool(x).squeeze(-1)
        x      = self.drop(self.gelu(self.fc1(x)))
        emb    = self.fc_emb(x)
        logits = self.classifier(emb)
        if return_emb and return_logits: return emb, logits
        if return_emb:                   return emb
        return logits
    

class MLP(nn.Module):
    """
    Deep MLP on hand-crafted features, single window, no temporal context.

    Input : (B, 8)  RMS per channel
    """
    def __init__(self, n_features: int = 8, emb_dim: int = 128,
                 num_classes: int = CLASSES, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 256),        nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),        nn.ReLU(), nn.Dropout(dropout),
        )
        self.fc_emb     = nn.Linear(128, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, return_emb=False, return_logits=False):
        # x: (B, 8)
        x      = self.net(x)
        emb    = self.fc_emb(x)
        logits = self.classifier(emb)
        if return_emb and return_logits: return emb, logits
        if return_emb:                   return emb
        return logits


# LSTM variants  
class LSTM(nn.Module):
    """
    LSTM over raw EMG timesteps within a single 200ms window.
    Processes 40 raw timesteps x 8 channels.

    Input : (B, 8, 40) ->  internally transposed to (B, 40, 8)
    """
    def __init__(self, ch: int = CH, hidden: int = 128, num_layers: int = 3,
                 emb_dim: int = 128, num_classes: int = CLASSES,
                 dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=ch, hidden_size=hidden, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc1        = nn.Linear(hidden, 128)
        self.fc_emb     = nn.Linear(128, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.drop       = nn.Dropout(dropout)
        self.gelu       = nn.GELU()
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, return_emb=False, return_logits=False):
        # x: (B, 8, 40)
        x = x / 128.0
        x = x.permute(0, 2, 1)           # (B, 40, 8)
        _, (h_n, _) = self.lstm(x)
        x      = h_n[-1]                 # (B, hidden)
        x      = self.drop(self.gelu(self.fc1(x)))
        emb    = self.fc_emb(x)
        logits = self.classifier(emb)
        if return_emb and return_logits: return emb, logits
        if return_emb:                   return emb
        return logits


class LSTM_HCF(nn.Module):
    """
    LSTM over sub-windowed RMS features within a single 200ms window.
    features: fine-grained temporal structure (4 x 50ms RMS steps)
    rather than a single vector.

    Input : (B, n_sub, 8)  sub-windowed RMS  (pre-computed, see extract_sub_rms)
            default n_sub=4 → 4 x 50ms steps
    """
    def __init__(self, n_features: int = 8, n_sub: int = N_SUB,
                 hidden: int = 128, num_layers: int = 3,
                 emb_dim: int = 128, num_classes: int = CLASSES,
                 dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc1        = nn.Linear(hidden, 128)
        self.fc_emb     = nn.Linear(128, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.drop       = nn.Dropout(dropout)
        self.gelu       = nn.GELU()
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, return_emb=False, return_logits=False):
        # x: (B, n_sub, 8)
        _, (h_n, _) = self.lstm(x)
        x      = h_n[-1]                 # (B, hidden)
        x      = self.drop(self.gelu(self.fc1(x)))
        emb    = self.fc_emb(x)
        logits = self.classifier(emb)
        if return_emb and return_logits: return emb, logits
        if return_emb:                   return emb
        return logits


class CNN(nn.Module):
    """
    Single-scale CNN on raw EMG, single window.
    Ablation of MHCNN: isolates the contribution of
    multi-horizon parallel dilation from raw signal access alone.

    Input : (B, 8, 40) 
    """
    def __init__(self, ch: int = CH, emb_dim: int = 128,
                 num_classes: int = CLASSES, dropout: float = DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv1d(ch,  32, kernel_size=4, padding='same')
        self.conv2 = nn.Conv1d(32,  128, kernel_size=3, padding='same')
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding='same')
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc1        = nn.Linear(128, 128)
        self.fc_emb     = nn.Linear(128, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, return_emb=False, return_logits=False):
        # x: (B, 8, 40)
        x = x / 128.0         # Scaling, not normalization
        x = self.drop(self.relu(self.conv1(x)))
        x = self.drop(self.relu(self.conv2(x)))
        x = self.drop(self.relu(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x      = self.drop(self.gelu(self.fc1(x)))
        emb    = self.fc_emb(x)
        logits = self.classifier(emb)
        if return_emb and return_logits: return emb, logits
        if return_emb:                   return emb
        return logits

class CNN_HCF(nn.Module):
    """
    1D CNN over sub-windowed hand-crafted features.
    Input : (B, F, 4)  — F features as channels, 4 sub-windows as time.
    """
    def __init__(self, n_feat: int, n_sub: int = N_SUB,
                 emb_dim: int = 128, num_classes: int = CLASSES,
                 dropout: float = DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv1d(n_feat, 64,  kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(64,    128,  kernel_size=3, padding='same')
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc1        = nn.Linear(128, 128)
        self.fc_emb     = nn.Linear(128, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, return_emb=False, return_logits=False):
        # x: (B, F, 4)
        x = self.drop(self.relu(self.conv1(x)))
        x = self.drop(self.relu(self.conv2(x)))
        x = self.pool(x).squeeze(-1)              # (B, 128)
        x      = self.drop(self.gelu(self.fc1(x)))
        emb    = self.fc_emb(x)
        logits = self.classifier(emb)
        if return_emb and return_logits: return emb, logits
        if return_emb:                   return emb
        return logits


# -------- GRL --------
class _GRLFn(Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class GRL(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x):
        return _GRLFn.apply(x, self.lambd)
    
class MHCNN_GRL(nn.Module):
    def __init__(self, ch=CH, seq=SEQ, emb_dim=128, 
                 num_classes=CLASSES, num_grl=306,
                 lambd=1.0, dropout=DROPOUT):
        super().__init__()
        self.emb_dim = emb_dim
        self.dropout = dropout

        self.conv1 = nn.Conv1d(ch, 32, 8, dilation=1, padding="same")
        self.conv2 = nn.Conv1d(ch, 32, 8, dilation=2, padding="same")
        self.conv3 = nn.Conv1d(ch, 32, 8, dilation=4, padding="same")
        self.conv4 = nn.Conv1d(96, 128, 4, dilation=1, padding="same")

        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(128, 128)
        self.fc_emb = nn.Linear(128, emb_dim)

        self.drop = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

        self.classifier = nn.Linear(self.emb_dim, 
                                    num_classes)
        
        self.grl = GRL(lambd=lambd)
        self.classifier_grl = nn.Linear(self.emb_dim, 
                                        num_grl)

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, return_emb=False, 
                return_logits=False, return_grl=False):
        x = x / 128.0         # Scaling, not normalization

        x1 = self.relu(self.conv1(x))
        x1 = self.drop(x1)
        x2 = self.relu(self.conv2(x))
        x2 = self.drop(x2)
        x3 = self.relu(self.conv3(x))
        x3 = self.drop(x3)
        x = torch.cat((x1, x2, x3), 1)
        x = self.relu(self.conv4(x))
        x = self.drop(x)

        x = self.pool(x).squeeze(-1)
        # x = x.flatten(1)

        x = self.fc1(x)
        x = self.gelu(x)
        emb = self.fc_emb(x)

        logits = self.classifier(emb)
        logits_grl = self.classifier_grl(self.grl(emb))

        if return_grl:
            logits = (logits, logits_grl)

        if return_emb and return_logits:
            return emb, logits
        if return_emb:
            return emb
        return logits
    

# ======== LOSSES ========

class BaseLoss(nn.Module):
    def __init__(self, weight=None,):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, labels, *args):
        return self.ce(logits, labels)
    

class RestLoss(nn.Module):
    def __init__(self, alpha1=1.0, alpha2=1.0, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none', 
                                      weight=weight)
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, logits, targets):
        loss = self.ce(logits, targets)
        pred = logits.argmax(1)

        p1 = (pred != targets) & (pred != 0)
        loss = loss * (1 + self.alpha1 * p1.float())

        p2 = (targets == 0) & (pred != 0)
        loss = loss * (1 + self.alpha2 * p2.float())

        return loss.mean()


class ActiveLoss(nn.Module):
    def __init__(self, alpha1=1.0, alpha2=0.1, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none', 
                                      weight=weight)
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, logits, targets):
        loss = self.ce(logits, targets)
        pred = logits.argmax(1)

        p1 = (targets != 0) | (pred != 0)
        loss = loss * (1 + self.alpha1 * p1.float())

        p2 = (targets == 0)
        loss = loss * (1 + self.alpha2 * p2.float())

        return loss.mean()


def _balanced_mean(loss, logits, targets):
    targets_flat = targets.flatten()
    losses_flat = loss.flatten()
    C = logits.size(1)
    device = logits.device
    class_loss_sum = torch.zeros(C, device=device).scatter_add(
        0, targets_flat, losses_flat)
    class_counts = torch.zeros(C, device=device).scatter_add(
        0, targets_flat, torch.ones_like(losses_flat))
    mask = class_counts > 0
    return (class_loss_sum[mask] / class_counts[mask]).mean()


class STDLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(reduction="none", weight=None)

    def forward(self, logits, targets):
        l = self.ce(logits, targets)
        mean = _balanced_mean(l, logits, targets)
        return mean + self.alpha * l.std(unbiased=False)


class PerSubjectLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none", weight=weight)

    def forward(self, logits, targets, users):
        loss = self.ce(logits, targets)  # (B,)
        unique, inverse = torch.unique(users, return_inverse=True)
        S = unique.size(0)
        device = logits.device

        loss_sum = torch.zeros(S, device=device).scatter_add(
            0, inverse, loss)
        counts = torch.zeros(S, device=device).scatter_add(
            0, inverse, torch.ones_like(loss))
        
        per_user = loss_sum / counts.clamp_min(1)
        
        return per_user.mean() + per_user.std(unbiased=False)
    

class CVaRLoss(nn.Module):
    def __init__(self, alpha=0.3, weight=None):
        super().__init__()
        assert 0 < alpha <= 1, "alpha must be in (0, 1]"
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(reduction='none', 
                                      weight=weight)

    def forward(self, logits, targets):
        # Per-sample cross entropy
        ce = self.ce(logits, targets)
        batch_size = ce.size(0)
        k = max(1, int(self.alpha * batch_size))
        # Select top-k highest losses
        topk_loss, _ = torch.topk(ce, k=k, largest=True)
        return topk_loss.mean()


class PrototypeLoss(nn.Module):
    def __init__(self, lambda_proto=0.5, normalize=True, weight=None):
        super().__init__()
        self.lambda_proto = lambda_proto
        self.normalize = normalize
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, emb, logits, labels):

        if self.normalize:
            emb = F.normalize(emb, dim=1)

        ce = self.ce(logits, labels)

        classes = torch.unique(labels)
        proto_loss = 0.0
        count = 0

        for c in classes:
            mask = labels == c
            z = emb[mask]

            if z.size(0) <= 1:
                continue

            proto = z.mean(dim=0, keepdim=True)
            proto_loss += ((z - proto) ** 2).sum(dim=1).mean()
            count += 1

        if count > 0:
            proto_loss = proto_loss / count

        loss = (1 - self.lambda_proto) * ce + self.lambda_proto * proto_loss

        return loss


class OneVsAllLoss(nn.Module):
    def __init__(self, lambda_proto=0.5, normalize=True, weight=None):
        super().__init__()
        self.lambda_proto = lambda_proto
        self.normalize = normalize
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, emb, logits, labels):
        if self.normalize:
            emb = F.normalize(emb, dim=1)

        ce = self.ce(logits, labels)
        classes = torch.unique(labels)
        proto_loss = 0.0

        if len(classes) == 0:
            pass
        elif len(classes) == 1:
            c = classes[0]
            mask = labels == c
            z = emb[mask]
            if z.size(0) > 1:
                proto = z.mean(dim=0, keepdim=True)
                proto_loss = ((z - proto) ** 2).sum(dim=1).mean()
        else:
            # Multiple classes → full prototype loss:
            #   - distance to own class mean  → lower loss when closer
            #   - distance to every other class mean → lower loss when farther
            # This is equivalent to a one-vs-all style softmax over negative distances
            # (standard prototypical / metric-learning loss)
            proto_list = []
            class_to_idx = {}
            for idx, c in enumerate(classes):
                mask = labels == c
                z = emb[mask]
                proto = z.mean(dim=0)                     # (D,)
                proto_list.append(proto)
                class_to_idx[c.item()] = idx

            protos = torch.stack(proto_list, dim=0)       # (K, D) where K = #classes in batch

            # Squared Euclidean distances from every embedding to every prototype
            # shape: (N, K)
            dists = ((emb.unsqueeze(1) - protos.unsqueeze(0)) ** 2).sum(dim=-1)

            target_idx = torch.tensor(
                [class_to_idx[l.item()] for l in labels],
                dtype=torch.long,
                device=labels.device)

            # Proto loss = CrossEntropy( -distances, true_class )
            # → minimizing this simultaneously:
            #     • pulls each sample toward its own class mean (closer = lower loss)
            #     • pushes each sample away from all other class means (farther = lower loss)
            proto_loss = F.cross_entropy(-dists, target_idx)

        loss = (1 - self.lambda_proto) * ce + self.lambda_proto * proto_loss

        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5, w_hard=1.0, w_soft=0.0,
                 batch_hard=True, normalize=True):
        super().__init__()
        self.margin = margin
        self.w_hard = w_hard
        self.w_soft = w_soft
        self.normalize = normalize
        self.bacth_hard = batch_hard
        self.triplet = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
            margin=margin,
            reduction="mean"
        )

    def _batch_random(self, z, pos_mask, neg_mask):
        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
 
        if valid.sum() == 0:
            return z.sum() * 0.0  # safe zero with grad
 
        p_idx = torch.multinomial(pos_mask[valid].float(), 1).squeeze(1)
        n_idx = torch.multinomial(neg_mask[valid].float(), 1).squeeze(1)

        a = z[valid]
        p = z[p_idx]
        n = z[n_idx]
 
        return self.triplet(a, p, n)

    def _batch_hard(self, z, pos_mask, neg_mask):
        dist = 1 - torch.matmul(z, z.T)

        pos_dist = dist.masked_fill(~pos_mask, float("-inf"))
        neg_dist = dist.masked_fill(~neg_mask, float("inf"))

        p_idx = pos_dist.argmax(dim=1)
        n_idx = neg_dist.argmin(dim=1)

        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)

        if valid.sum() == 0:
            return z.sum() * 0.0  # safe zero with grad

        a = z[valid]
        p = z[p_idx[valid]]
        n = z[n_idx[valid]]

        return self.triplet(a, p, n)

    def batch_helper(self, *args, **kwargs):
        return self._batch_hard(*args, **kwargs) if self.bacth_hard \
            else self._batch_random(*args, **kwargs)

    def forward(self, z, labels, subjects):
        if self.normalize:
            z = F.normalize(z, dim=1)

        labels = labels.unsqueeze(1)
        subjects = subjects.unsqueeze(1)

        same_class = labels == labels.T
        diff_class = ~same_class
        same_subj  = subjects == subjects.T
        diff_subj  = ~same_subj

        N = labels.size(0)
        eye = torch.eye(N, dtype=torch.bool, device=z.device)

        pos_hard = same_class & diff_subj & ~eye
        neg_hard = diff_class & same_subj & ~eye

        loss = torch.tensor(0.0, device=z.device)
        denom = 0

        if self.w_hard != 0:
            loss += self.w_hard * self.batch_helper(z, pos_hard, neg_hard)
            denom += abs(self.w_hard)

        if self.w_soft != 0:
            pos_soft = same_class & same_subj & ~eye
            neg_soft = diff_class & diff_subj & ~eye
            loss += self.w_soft * self.batch_helper(z, pos_soft, neg_soft)
            denom += abs(self.w_soft)

        return loss / denom



class AngularLoss(nn.Module):
    def __init__(self, temperature=0.07, normalize=True, weight=None, 
                 nm_label=0, w_ce=1.0, w_supcon=1.0, w_angle=1.0):
        super().__init__()
        self.tau = temperature
        self.normalize = normalize
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.nm_label = nm_label # Explicitly define which label is No Motion
        
        # Add weights to balance the loss scales
        self.w_ce = w_ce
        self.w_supcon = w_supcon
        self.w_angle = w_angle

    def forward(self, emb, logits, labels, *args):
        if self.normalize:
            emb = F.normalize(emb, dim=1)

        ce_loss = self.ce(logits, labels)
        classes = torch.unique(labels)
        angle_loss = torch.tensor(0.0, device=emb.device)

        if len(classes) > 1:
            proto_list = []
            class_to_idx = {}
            for idx, c in enumerate(classes):
                mask = labels == c
                z = emb[mask]
                proto = z.mean(dim=0)
                proto_list.append(proto)
                class_to_idx[c.item()] = idx

            protos = torch.stack(proto_list, dim=0)

            # 1. FIX: Safely find the NM prototype
            if self.nm_label in class_to_idx:
                nm_idx = class_to_idx[self.nm_label]
                nm_proto = protos[nm_idx]
                
                # Extract only the active classes (exclude NM)
                active_mask = torch.arange(len(protos)) != nm_idx
                active_protos = protos[active_mask]

                # 2. FIX: Correctly calculate angular penalty on valid upper triangle
                if len(active_protos) > 1:
                    directions = F.normalize(active_protos - nm_proto, dim=1)
                    cos_sim = directions @ directions.T
                    
                    # Extract only the strictly upper triangle elements (no zeros included)
                    triu_indices = torch.triu_indices(row=cos_sim.size(0), col=cos_sim.size(1), offset=1)
                    angle_loss = cos_sim[triu_indices[0], triu_indices[1]].mean()

        # --- Supervised Contrastive Loss ---
        sim = emb @ emb.T / self.tau
        N = emb.size(0)
        mask_self = ~torch.eye(N, dtype=torch.bool, device=emb.device)
        
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & mask_self

        # 3. FIX: Numerical stability for exponents
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim_stable = sim - sim_max.detach()

        exp_sim = torch.exp(sim_stable) * mask_self
        log_prob = sim_stable - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        pos_count = pos_mask.sum(dim=1).clamp(min=1)
        supcon_loss = -(log_prob * pos_mask).sum(dim=1) / pos_count
        supcon_loss = supcon_loss.mean()

        # Weighted final loss
        total_loss = (self.w_ce * ce_loss) + (self.w_supcon * supcon_loss) + (self.w_angle * angle_loss)

        return total_loss