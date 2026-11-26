"""Pコン跡 classifier with neighbourhood context.

Each 256px crop is looked at together with a margin taken from the re-assembled source photo,
plus a 4th input channel marking which part is the crop itself. A hole cut by the crop border
(which the rules count as "contained") is then seen whole. Trained from scratch (no ImageNet).
CV: StratifiedGroupKFold by source photo; 8-way dihedral TTA; F1-optimal threshold on OOF.

  python pcone/train_ctx.py --root data/sdnet --mosaic data/mosaic_half --train T.csv --test t.txt \
      --out runs/ctx_small --arch small --margin 32
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from data import read_list  # noqa: E402
from mosaic import TILE, cell, parse  # noqa: E402
from train_cv import Block, best_threshold, dihedral, get_device  # noqa: E402


class SmallNet(nn.Module):
    def __init__(self, in_ch=4, w=32, drop=0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, w, 3, 2, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(True))
        layers, ci = [], w
        for i, co in enumerate([w, w * 2, w * 4, w * 8]):
            layers += [Block(ci, co, 1 if i == 0 else 2), Block(co, co, 1)]
            ci = co
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(ci, 1))

    def forward(self, x):
        x = self.body(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


def build(arch, in_ch=4):
    if arch.startswith("small"):
        w = int(arch[5:] or 32)
        return SmallNet(in_ch, w)
    m = getattr(torchvision.models, arch)(weights=None, num_classes=1)
    m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    return m


class Windows:
    """Extract (crop + margin) windows from padded mosaics; centre = the listed crop."""

    def __init__(self, root, mosaic_dir, margin):
        self.root, self.dir, self.m = Path(root), Path(mosaic_dir), margin
        self.cache = {}
        probe = np.load(next(self.dir.glob("[0-9]*[0-9].npy")), mmap_mode="r")
        self.t = probe.shape[0] // 14
        self.size = self.t + 2 * margin

    def _mosaic(self, pid):
        if pid not in self.cache:
            p = self.dir / f"{pid}.npy"
            M = np.load(p) if p.exists() else None
            if M is not None:
                M = np.pad(M, ((self.m, self.m), (self.m, self.m), (0, 0)), constant_values=128)
            self.cache[pid] = M
        return self.cache[pid]

    def _crop(self, rel):
        img = cv2.cvtColor(cv2.imread(str(self.root / rel)), cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (self.t, self.t), interpolation=cv2.INTER_AREA) if self.t != TILE else img

    def get(self, rel):
        p = parse(rel)
        M = self._mosaic(p[0]) if p else None
        if M is None or not (1 <= p[1] <= 252):
            w = np.full((self.size, self.size, 3), 128, np.uint8)
            w[self.m:self.m + self.t, self.m:self.m + self.t] = self._crop(rel)
            return w
        r, c = cell(p[1])
        t = self.t
        w = M[r * t:r * t + self.size, c * t:c * t + self.size].copy()
        if p[2] is not None:  # '_2' variants: use the actual file for the centre
            w[self.m:self.m + t, self.m:self.m + t] = self._crop(rel)
        return w

    def stack(self, rels):
        return np.stack([self.get(r) for r in rels])


def center_mask(size, m, device):
    k = torch.zeros(1, 1, size, size, device=device)
    k[..., m:size - m, m:size - m] = 1.0
    return k


MEAN, STD = 0.5, 0.25


def to_input(xb, mask):
    x = xb.permute(0, 3, 1, 2).float() / 255.0
    x = (x - MEAN) / STD
    return torch.cat([x, mask.expand(x.shape[0], 1, -1, -1)], 1)


def augment(x):
    """x: (B,4,H,W) incl. mask channel (dihedral-invariant). Photometric jitter on RGB only."""
    B = x.shape[0]
    x = dihedral(x, np.random.randint(8))
    rgb, m = x[:, :3], x[:, 3:]
    a = 1 + 0.25 * (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1)
    b = 0.25 * (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1)
    t = 0.08 * (torch.rand(B, 3, 1, 1, device=x.device) * 2 - 1)
    rgb = rgb * a + b + t
    g = (torch.rand(B, 1, 1, 1, device=x.device) < 0.2).float()
    rgb = g * rgb.mean(1, keepdim=True).expand_as(rgb) + (1 - g) * rgb
    rgb = rgb + 0.05 * torch.randn_like(rgb) * (torch.rand(B, 1, 1, 1, device=x.device) < 0.3).float()
    return torch.cat([rgb, m], 1)


@torch.no_grad()
def predict(model, X, mask, device, bs=256, tta=8):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = to_input(torch.from_numpy(X[i:i + bs]).to(device), mask)
        p = 0
        for k in range(tta):
            p = p + torch.sigmoid(model(dihedral(x, k))).squeeze(1)
        out.append((p / tta).float().cpu().numpy())
    return np.concatenate(out)


def train_fold(Xtr, ytr, Xva, yva, mask, a, device, fold, log):
    model = build(a.arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * a.steps,
                                                pct_start=0.1)
    pos = ytr.mean()
    w = np.where(ytr == 1, a.pos_frac / max(pos, 1e-6), (1 - a.pos_frac) / max(1 - pos, 1e-6))
    w = w / w.sum()
    yt = torch.from_numpy(ytr.astype(np.float32))
    hist = []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); losses = []
        for _ in range(a.steps):
            idx = np.random.choice(len(Xtr), a.bs, p=w)
            x = augment(to_input(torch.from_numpy(Xtr[idx]).to(device), mask))
            y = yt[idx].to(device)
            loss = F.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        if ep % a.eval_every == 0 or ep == a.epochs:
            p = predict(model, Xva, mask, device, tta=1)
            th, f1 = best_threshold(yva, p)
            hist.append({"ep": ep, "loss": float(np.mean(losses)), "f1": f1, "th": th})
            print(f"[fold{fold}] ep{ep:3d} loss={np.mean(losses):.4f} valF1={f1:.4f}@{th:.2f} "
                  f"({time.time() - t0:.0f}s/ep)", flush=True, file=log)
    return model, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--mosaic", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=int, default=32, help="context margin in mosaic pixels")
    ap.add_argument("--arch", default="small")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--run-folds", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps", type=int, default=150, help="batches per epoch")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--pos-frac", type=float, default=0.35)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = sys.stdout
    device = get_device()
    tr = read_list(a.train); te = read_list(a.test)
    tr_paths = [r[0] for r in tr]; y = np.array([int(r[1]) for r in tr]); te_paths = [r[0] for r in te]
    groups = np.array([(parse(p) or (p,))[0] for p in tr_paths])
    W = Windows(a.root, a.mosaic, a.margin)
    t0 = time.time()
    X = W.stack(tr_paths); Xte = W.stack(te_paths)
    print(f"device={device} train={len(y)} pos={y.sum()} ({y.mean():.2%}) photos={len(set(groups))} "
          f"test={len(te_paths)} window={W.size}px (loaded {time.time() - t0:.0f}s)", flush=True)
    mask = center_mask(W.size, a.margin, device)

    skf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
    oof = np.full(len(y), np.nan); test_prob = np.zeros(len(te_paths)); n_run = 0; hists = {}
    for fold, (tri, vai) in enumerate(skf.split(X, y, groups)):
        if a.run_folds and fold >= a.run_folds:
            break
        model, hist = train_fold(X[tri], y[tri], X[vai], y[vai], mask, a, device, fold, log)
        oof[vai] = predict(model, X[vai], mask, device)
        test_prob += predict(model, Xte, mask, device); n_run += 1; hists[fold] = hist
        th, f1 = best_threshold(y[vai], oof[vai])
        print(f"[fold{fold}] TTA F1={f1:.4f}@{th:.2f}", flush=True)
        torch.save(model.state_dict(), out / f"model_f{fold}.pt")
    test_prob /= n_run
    m = ~np.isnan(oof)
    th, f1 = best_threshold(y[m], oof[m])
    print(f"OOF F1={f1:.4f} at th={th:.2f} (n={m.sum()})", flush=True)
    np.save(out / "oof.npy", oof); np.save(out / "test_prob.npy", test_prob)
    pred = (test_prob >= th).astype(int)
    with open(out / "submission.csv", "w", newline="\n") as fh:
        fh.writelines(f"{p},{v}\n" for p, v in zip(te_paths, pred))
    json.dump({"args": vars(a), "oof_f1": f1, "threshold": th, "test_pos": int(pred.sum()),
               "hist": hists}, open(out / "report.json", "w"), indent=1)
    print(f"submission -> {out/'submission.csv'}  predicted positives={pred.sum()}/{len(pred)}")


if __name__ == "__main__":
    main()
