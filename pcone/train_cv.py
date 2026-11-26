"""Pコン跡 classifier: CNN trained from scratch (no ImageNet weights, per contest rules),
StratifiedGroupKFold by source photo, 8-way dihedral TTA, F1-optimal threshold on OOF.

  python pcone/train_cv.py --root DATA/SDNET2018 --train DATA/train_label.csv --test DATA/test.txt \
      --out runs/pc_r18_128 --size 128 --arch resnet18 --epochs 30
Writes <out>/oof.npy, <out>/test_prob.npy, <out>/submission.csv, <out>/report.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from data import load_images, photo_id, read_list  # noqa: E402


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------ models (random init only)
class Block(nn.Module):
    def __init__(self, ci, co, s):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, s, 1, bias=False); self.b1 = nn.BatchNorm2d(co)
        self.c2 = nn.Conv2d(co, co, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(co)
        self.sc = None if (ci == co and s == 1) else nn.Sequential(
            nn.Conv2d(ci, co, 1, s, bias=False), nn.BatchNorm2d(co))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.sc is None else self.sc(x)))


class SmallNet(nn.Module):
    def __init__(self, w=32, drop=0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 2, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(True))
        chans = [w, w * 2, w * 4, w * 8]
        layers, ci = [], w
        for i, co in enumerate(chans):
            layers += [Block(ci, co, 1 if i == 0 else 2), Block(co, co, 1)]
            ci = co
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(ci, 1))

    def forward(self, x):
        x = self.body(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


def build(arch):
    if arch == "small":
        return SmallNet(32)
    if arch == "small48":
        return SmallNet(48)
    m = getattr(torchvision.models, arch)(weights=None, num_classes=1)
    return m


# ------------------------------------------------------------------ GPU-side augmentation
MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
STD = torch.tensor([0.25, 0.25, 0.25]).view(1, 3, 1, 1)


def to_input(xb_uint8, device):
    x = xb_uint8.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN.to(device)) / STD.to(device)


def dihedral(x, k):
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, k % 4, dims=(-2, -1))


def augment(x):
    B = x.shape[0]
    k = np.random.randint(8)
    x = dihedral(x, k)
    # brightness / contrast / per-channel tint jitter
    a = 1 + 0.25 * (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1)
    b = 0.25 * (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1)
    t = 0.08 * (torch.rand(B, 3, 1, 1, device=x.device) * 2 - 1)
    x = x * a + b + t
    # random grayscale
    g = (torch.rand(B, 1, 1, 1, device=x.device) < 0.2).float()
    x = g * x.mean(1, keepdim=True).expand_as(x) + (1 - g) * x
    # small random zoom (0.9-1.1) without cropping marks away: scale about centre with reflection
    if np.random.rand() < 0.5:
        s = float(np.random.uniform(0.9, 1.1))
        theta = torch.tensor([[s, 0, 0], [0, s, 0]], device=x.device, dtype=x.dtype).expand(B, 2, 3)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    # gaussian noise
    x = x + 0.05 * torch.randn_like(x) * (torch.rand(B, 1, 1, 1, device=x.device) < 0.3).float()
    return x


@torch.no_grad()
def predict(model, X, device, bs=256, tta=8):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = to_input(torch.from_numpy(X[i:i + bs]), device)
        p = 0
        for k in range(tta):
            p = p + torch.sigmoid(model(dihedral(x, k))).squeeze(1)
        out.append((p / tta).float().cpu().numpy())
    return np.concatenate(out)


def best_threshold(y, p):
    ths = np.linspace(0.05, 0.95, 91)
    f = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in ths]
    i = int(np.argmax(f))
    return float(ths[i]), float(f[i])


def train_fold(Xtr, ytr, Xva, yva, a, device, fold):
    model = build(a.arch).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = int(np.ceil(len(Xtr) / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch,
                                                pct_start=0.1)
    pos = ytr.mean()
    # balanced-ish sampling: positives drawn with prob a.pos_frac per batch slot
    w = np.where(ytr == 1, a.pos_frac / max(pos, 1e-6), (1 - a.pos_frac) / max(1 - pos, 1e-6))
    w = w / w.sum()
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr.astype(np.float32))
    best, best_state, hist = -1, None, []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); losses = []
        for _ in range(steps_per_epoch):
            idx = np.random.choice(len(Xtr), a.bs, p=w)
            x = augment(to_input(Xt[idx], device)).contiguous(memory_format=torch.channels_last)
            y = yt[idx].to(device)
            if a.label_smooth:
                y = y * (1 - a.label_smooth) + 0.5 * a.label_smooth
            loss = F.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        if ep % a.eval_every == 0 or ep == a.epochs:
            p = predict(model, Xva, device, tta=1)
            th, f1 = best_threshold(yva, p)
            hist.append({"ep": ep, "loss": float(np.mean(losses)), "f1": f1, "th": th})
            print(f"[fold{fold}] ep{ep:3d} loss={np.mean(losses):.4f} valF1={f1:.4f}@{th:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if f1 >= best:
                best, best_state = f1, {k: v.detach().clone() for k, v in model.state_dict().items()}
    if a.keep == "best" and best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="SDNET2018 folder that contains W/")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--run-folds", type=int, default=0, help="train only the first N folds (0=all)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--pos-frac", type=float, default=0.35)
    ap.add_argument("--label-smooth", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--keep", choices=["best", "last"], default="last")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    tr_rows = read_list(a.train)
    te_rows = read_list(a.test)
    tr_paths = [r[0] for r in tr_rows]
    y = np.array([int(r[1]) for r in tr_rows])
    te_paths = [r[0] for r in te_rows]
    groups = np.array([photo_id(p) for p in tr_paths])
    print(f"device={device} train={len(y)} pos={y.sum()} ({y.mean():.3%}) photos={len(set(groups))} "
          f"test={len(te_paths)}", flush=True)

    X = load_images(a.root, tr_paths, a.size, Path(a.cache) / f"train_{a.size}.npy")
    Xte = load_images(a.root, te_paths, a.size, Path(a.cache) / f"test_{a.size}.npy")

    skf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
    oof = np.full(len(y), np.nan)
    test_prob = np.zeros(len(te_paths))
    n_run, hists = 0, {}
    for fold, (tri, vai) in enumerate(skf.split(X, y, groups)):
        if a.run_folds and fold >= a.run_folds:
            break
        model, hist = train_fold(X[tri], y[tri], X[vai], y[vai], a, device, fold)
        oof[vai] = predict(model, X[vai], device)
        test_prob += predict(model, Xte, device)
        n_run += 1; hists[fold] = hist
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
        for p, v in zip(te_paths, pred):
            fh.write(f"{p},{v}\n")
    json.dump({"args": vars(a), "oof_f1": f1, "threshold": th, "test_pos_rate": float(pred.mean()),
               "hist": hists}, open(out / "report.json", "w"), indent=1)
    print(f"submission -> {out/'submission.csv'}  predicted positives={pred.sum()}/{len(pred)}")


if __name__ == "__main__":
    main()
