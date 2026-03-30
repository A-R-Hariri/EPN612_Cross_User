import torch
import torch.nn as nn; 
import torch.nn.functional as F
from torch.autograd import Function

from utils import *

    
# ======== MODELS ========
class MLP(nn.Module):
    def __init__(self, feats, emb_dim=128, proj_dim=128, dropout=DROPOUT):
        super().__init__()

        self.fc1 = nn.Linear(feats, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_emb = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, CLASSES)  # embedding
        
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, return_emb=False, return_logits=False):
        x = x / 128.0

        x = self.relu(self.fc1(x))
        x = self.drop(x)

        x = self.relu(self.fc2(x))
        x = self.drop(x)

        x = self.relu(self.fc3(x))
        x = self.drop(x)

        emb = self.relu(self.fc_emb(x))

        logits = self.classifier(emb)

        if return_emb and return_logits:
            return emb, logits
        if return_emb:
            return emb
        return logits
    

class CNNBasic(nn.Module):
    def __init__(self, ch=CH, num_classes=CLASSES, dropout=DROPOUT):
        super().__init__()

        self.conv1 = nn.Conv1d(ch, 32, 4, padding="same")
        self.conv2 = nn.Conv1d(32, 64, 4, padding="same")
        self.conv3 = nn.Conv1d(64, 96, 4, padding="same")
        self.conv4 = nn.Conv1d(96, 128, 4, padding="same")

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128, 128)
        self.fc_emb = nn.Linear(128, 128)

        self.classifier = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x / 128.0

        x = self.drop(self.relu(self.conv1(x)))
        x = self.drop(self.relu(self.conv2(x)))
        x = self.drop(self.relu(self.conv3(x)))
        x = self.drop(self.relu(self.conv4(x)))

        x = self.pool(x).squeeze(-1)

        x = self.relu(self.fc1(x))
        emb = self.fc_emb(x)

        logits = self.classifier(emb)
        return logits
    

class LSTM(nn.Module):
    def __init__(self, ch=CH, hidden=128, layers=2, num_classes=CLASSES):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=ch,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc1 = nn.Linear(hidden * 2, 128)
        self.fc_emb = nn.Linear(128, 128)

        self.classifier = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x / 128.0

        # x: (B, C, T) → (B, T, C)
        x = x.transpose(1,2)

        x, _ = self.lstm(x)

        x = x[:, -1]
        x = self.relu(self.fc1(x))
        emb = self.fc_emb(x)

        logits = self.classifier(emb)

        return logits
    

class TransformerEMG(nn.Module):
    def __init__(self, ch=CH, seq=SEQ, d_model=128, num_heads=4, num_layers=2, num_classes=CLASSES):
        super().__init__()

        self.proj = nn.Linear(ch, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            batch_first=True)

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers)

        self.fc1 = nn.Linear(128, 128)
        self.fc_emb = nn.Linear(128, 128)

        self.classifier = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x / 128.0

        # (B,C,T) -> (B,T,C)
        x = x.transpose(1,2)
        x = self.proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)

        x = self.relu(self.fc1(x))
        emb = self.fc_emb(x)

        logits = self.classifier(emb)

        return logits
    

# -------- Proposed --------
class CNN(nn.Module):
    def __init__(self, ch=CH, seq=SEQ, emb_dim=128, 
                 num_classes=CLASSES, dropout=DROPOUT):
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

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, return_emb=False, return_logits=False):
        x = x / 128.0

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

        if return_emb and return_logits:
            return emb, logits
        if return_emb:
            return emb
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
    
class CNN_GRL(nn.Module):
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
        x = x / 128.0

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
class RestLoss(nn.Module):
    def __init__(self, alpha1=0.25, alpha2=0.5, weight=None):
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


class EqLoss(nn.Module):
    def __init__(self, alpha=0.3, eps=1e-8, weight=None):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(reduction="none", weight=weight)

    def forward(self, logits, targets):

        l = self.ce(logits, targets)
        mean = l.mean()

        classes = torch.unique(targets)
        class_means = []

        for c in classes:
            mask = targets == c
            class_means.append(l[mask].mean())

        class_means = torch.stack(class_means)

        equity = class_means.var(unbiased=False) / (class_means.mean() + self.eps)

        return mean + self.alpha * equity
    

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


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3, w_hard=1.0, w_soft=0.0, normalize=True):
        super().__init__()
        self.margin = margin
        self.w_hard = w_hard
        self.w_soft = w_soft
        self.normalize = normalize
        self.triplet = nn.TripletMarginLoss(
            margin=margin, p=2, reduction="mean")

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
            loss += self.w_hard * self._batch_hard(z, pos_hard, neg_hard)
            denom += abs(self.w_hard)

        if self.w_soft != 0:
            pos_soft = same_class & same_subj & ~eye
            neg_soft = diff_class & diff_subj & ~eye
            loss += self.w_soft * self._batch_hard(z, pos_soft, neg_soft)
            denom += abs(self.w_soft)

        return loss / denom


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