"""Transformer encoder kecil untuk klasifikasi citra pintu 224x224.

    python models/123450029/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name
JALUR = "trf"
JUMLAH_KELAS = 4
EPOCH = 8
BATCH = 16
LR = 3e-4
BATAS_DETIK = 18 * 60


def bangun_model():
    import torch
    from torch import nn

    class TransformerCitra(nn.Module):
        def __init__(self):
            super().__init__()
            dim = 64
            # 224x224 dibagi menjadi patch 16x16: 14x14 token citra.
            self.patch = nn.Conv2d(3, dim, kernel_size=16, stride=16)
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos = nn.Parameter(torch.zeros(1, 197, dim))
            lapis = nn.TransformerEncoderLayer(
                d_model=dim, nhead=4, dim_feedforward=128,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(lapis, num_layers=2, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim, JUMLAH_KELAS))
            nn.init.trunc_normal_(self.pos, std=0.02)
            nn.init.trunc_normal_(self.cls, std=0.02)

        def forward(self, x):
            if x.ndim != 4 or tuple(x.shape[1:]) != (3, 224, 224):
                raise ValueError(f"Masukan model harus [N,3,224,224], didapat {tuple(x.shape)}")
            token = self.patch(x).flatten(2).transpose(1, 2)
            token = torch.cat((self.cls.expand(x.shape[0], -1, -1), token), dim=1)
            token = self.encoder(token + self.pos)
            logits = self.head(self.norm(token[:, 0]))
            if logits.shape[1] != JUMLAH_KELAS:
                raise ValueError(f"Keluaran harus memiliki {JUMLAH_KELAS} kelas, didapat {logits.shape[1]}")
            return logits

    return TransformerCitra()


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    Xtr, ytr, Xva, yva, Xte, _ = data.muat()
    if Xtr.ndim != 4 or tuple(Xtr.shape[1:]) != (3, 224, 224):
        raise ValueError(f"Bentuk X latih harus [N,3,224,224], didapat {Xtr.shape}")
    if Xva.ndim != 4 or tuple(Xva.shape[1:]) != (3, 224, 224):
        raise ValueError(f"Bentuk X validasi harus [N,3,224,224], didapat {Xva.shape}")
    if not set(np.unique(ytr)).issubset(set(range(JUMLAH_KELAS))):
        raise ValueError(f"Label latih harus berada pada rentang 0–{JUMLAH_KELAS - 1}")

    device = torch.device("cpu")
    model = bangun_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    train_set = TensorDataset(
        torch.from_numpy(np.asarray(Xtr, dtype=np.float32)),
        torch.from_numpy(np.asarray(ytr, dtype=np.int64)),
    )
    loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, generator=generator)
    x_val = torch.from_numpy(np.asarray(Xva, dtype=np.float32)).to(device)
    y_val = torch.from_numpy(np.asarray(yva, dtype=np.int64)).to(device)

    best_loss = float("inf")
    best_state = None
    t0 = time.time()
    for epoch in range(EPOCH):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val).item())
        print(f"epoch {epoch + 1}/{EPOCH}  val_loss={val_loss:.4f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if time.time() - t0 > BATAS_DETIK:
            print("Batas waktu tercapai; memakai checkpoint validasi terbaik.", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Pelatihan tidak menghasilkan checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_val = model(x_val).argmax(dim=1).cpu().numpy().astype(np.int64)
        pred_test = None
        if final and Xte is not None and len(Xte):
            x_test = torch.from_numpy(np.asarray(Xte, dtype=np.float32)).to(device)
            pred_test = model(x_test).argmax(dim=1).cpu().numpy().astype(np.int64)
    return pred_val, pred_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="gunakan hanya saat evaluasi uji sprint terakhir")
    ap.add_argument("--perangkat", default="cpu laptop")
    args = ap.parse_args()

    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(args.seed, args.final)
    durasi = time.time() - t0
    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", pred_val)
    cmd = [
        sys.executable, "-m", "src.common.evaluate", "--nim", NIM,
        "--model", JALUR, "--seed", str(args.seed),
        "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{durasi:.1f}",
        "--perangkat", args.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", pred_test)
        cmd += ["--pred-test", str(out / "pred_test.npy")]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(cmd, check=True, cwd=AKAR, env=env)


if __name__ == "__main__":
    main()
