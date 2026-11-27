
import json
import time
from pathlib import Path

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.protokol import data as data_loader
from src.common import manifest, split


SEED = 1
N_POHON = 300
FOLDER_HASIL = Path("results/baseline")


def ekstrak_fitur(X):
    # Kembalikan citra ke RGB 0-1.
    X = data_loader.balik_normalisasi(X)
    X = np.clip(X.transpose(0, 2, 3, 1), 0, 1)

    seluruh_fitur = []

    for citra in X:
        # 1. Histogram warna RGB (16 bin per kanal).
        fitur_hist = []
        for kanal in range(3):
            hist, _ = np.histogram(
                citra[:, :, kanal],
                bins=16,
                range=(0, 1)
            )
            hist = hist.astype(np.float32)
            hist /= max(hist.sum(), 1)
            fitur_hist.extend(hist)

        # 2. Histogram of Oriented Gradients.
        grayscale = rgb2gray(citra)
        fitur_hog = hog(
            grayscale,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True
        )

        seluruh_fitur.append(
            np.concatenate([
                np.asarray(fitur_hist, dtype=np.float32),
                fitur_hog.astype(np.float32)
            ])
        )

    return np.asarray(seluruh_fitur, dtype=np.float32)


def main():
    FOLDER_HASIL.mkdir(parents=True, exist_ok=True)

    print("Memuat data primer Tim 11...")
    X_train, y_train, X_val, y_val, _, _ = data_loader.muat()

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)

    print("\nEkstraksi Histogram warna + HOG...")
    awal_ekstraksi = time.perf_counter()

    fitur_train = ekstrak_fitur(X_train)
    fitur_val = ekstrak_fitur(X_val)

    waktu_ekstraksi = time.perf_counter() - awal_ekstraksi

    print("Fitur train:", fitur_train.shape)
    print("Fitur validation:", fitur_val.shape)
    print(f"Waktu ekstraksi: {waktu_ekstraksi:.2f} detik")

    print("\nMelatih Random Forest...")
    model = RandomForestClassifier(
        n_estimators=N_POHON,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1
    )

    awal_latih = time.perf_counter()
    model.fit(fitur_train, y_train)
    waktu_latih = time.perf_counter() - awal_latih

    print(f"Waktu pelatihan: {waktu_latih:.2f} detik")

    print("\nEvaluasi validation set...")
    pred_val = model.predict(fitur_val)

    acc = accuracy_score(y_val, pred_val)
    macro_f1 = f1_score(
        y_val,
        pred_val,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(y_val, pred_val, labels=[0, 1, 2, 3])

    # Simpan prediksi dan confusion matrix.
    np.save(FOLDER_HASIL / "pred_val.npy", pred_val)
    np.savetxt(
        FOLDER_HASIL / "confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d"
    )

    # Skema metrics mengikuti evaluator proyek.
    hasil = {
        "metrik": {
            "val_acc": round(float(acc), 6),
            "val_macro_f1": round(float(macro_f1), 6)
        },
        "seed": SEED,
        "data_hash": manifest.data_hash(
            f"split:{split.SEED_SPLIT}".encode()
        ),
        "perangkat": "cpu laptop",
        "model": "baseline",
        "n_val": int(len(y_val)),
        "n_test": 0,
        "durasi_detik": round(
            float(waktu_ekstraksi + waktu_latih), 4
        )
    }

    with open(FOLDER_HASIL / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(hasil, f, indent=2, ensure_ascii=False)

    print("\n===== HASIL BASELINE =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nMetrics tersimpan:", FOLDER_HASIL / "metrics.json")


if __name__ == "__main__":
    main()
