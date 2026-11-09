"""Kerangka run.py anggota. Salin folder ini menjadi models/<nim>/ lalu isi bagian MODEL.

    python models/<nim>/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data

NIM = Path(__file__).resolve().parent.name
JALUR = "ae"


def siapkan_citra(X):
    X = np.asarray(X, dtype=np.float32)

    if X.ndim == 3:
        X = X[:, None, :, :]
    elif X.ndim == 4 and X.shape[-1] in (1, 3, 4):
        X = X.transpose(0, 3, 1, 2)

    if X.ndim != 4:
        raise ValueError(f"Bentuk citra tidak sesuai: {X.shape}")

    if X.shape[1] not in (1, 3, 4):
        raise ValueError(f"Jumlah channel tidak didukung: {X.shape}")

    if X.max() > 1:
        X = X / 255.0

    return torch.tensor(X, dtype=torch.float32)


class Autoencoder(nn.Module):
    def __init__(self, channels, jumlah_kelas):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 96, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(96, 128, 4, 2, 1),
            nn.ReLU()
        )

        self.latent = nn.Linear(128 * 7 * 7, 128)

        self.classifier = nn.Linear(128, jumlah_kelas)

        self.decoder_input = nn.Linear(128, 128 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 96, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        fitur = self.encoder(x)
        fitur = fitur.flatten(1)
        z = self.latent(fitur)

        rekonstruksi = self.decoder_input(z)
        rekonstruksi = rekonstruksi.reshape(-1, 128, 7, 7)
        rekonstruksi = self.decoder(rekonstruksi)

        prediksi = self.classifier(z)

        return rekonstruksi, prediksi


def latih_dan_prediksi(seed: int, final: bool):
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr, Xva, yva, Xte, _ = data.muat()

    X_train = siapkan_citra(Xtr)
    X_val = siapkan_citra(Xva)

    y_train = torch.tensor(np.asarray(ytr), dtype=torch.long)
    y_val = torch.tensor(np.asarray(yva), dtype=torch.long)

    jumlah_kelas = len(np.unique(y_train.numpy()))
    channels = X_train.shape[1]

    model = Autoencoder(channels, jumlah_kelas)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_rekonstruksi = nn.MSELoss()
    loss_klasifikasi = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(20):
        model.train()

        for citra, label in loader:
            optimizer.zero_grad()

            hasil_rekonstruksi, prediksi = model(citra)

            loss_ae = loss_rekonstruksi(hasil_rekonstruksi, citra)
            loss_cls = loss_klasifikasi(prediksi, label)

            loss = loss_ae + loss_cls

            loss.backward()
            optimizer.step()

    model.eval()

    with torch.no_grad():
        _, hasil_val = model(X_val)
        pred_val = hasil_val.argmax(1).numpy()

        pred_test = None

        if final and Xte is not None:
            X_test = siapkan_citra(Xte)
            _, hasil_test = model(X_test)
            pred_test = hasil_test.argmax(1).numpy()

    return pred_val, pred_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--perangkat", default="cpu laptop")
    args = parser.parse_args()

    waktu_awal = time.time()

    pred_val, pred_test = latih_dan_prediksi(args.seed, args.final)

    durasi = time.time() - waktu_awal

    output = AKAR / "results" / NIM
    output.mkdir(parents=True, exist_ok=True)

    np.save(output / "pred_val.npy", pred_val)

    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(args.seed),
        "--pred-val", str(output / "pred_val.npy"),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", args.perangkat
    ]

    if pred_test is not None:
        np.save(output / "pred_test.npy", pred_test)
        perintah += ["--pred-test", str(output / "pred_test.npy")]

    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
