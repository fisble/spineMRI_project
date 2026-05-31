"""
split.py  —  Patient-level 80/10/10 train/val/test split.
Keeps all slices of one patient in the same split (no data leakage).
"""

import random
from collections import defaultdict
from dataset import build_records


def split_records(foramina_dir: str, seed: int = 42):
    records = build_records(foramina_dir)

    # Group by patient
    by_patient = defaultdict(list)
    for r in records:
        by_patient[r["patient_id"]].append(r)

    patients = sorted(by_patient.keys())
    random.seed(seed)
    random.shuffle(patients)

    n = len(patients)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    train_pts = set(patients[:n_train])
    val_pts   = set(patients[n_train:n_train + n_val])
    test_pts  = set(patients[n_train + n_val:])

    train = [r for r in records if r["patient_id"] in train_pts]
    val   = [r for r in records if r["patient_id"] in val_pts]
    test  = [r for r in records if r["patient_id"] in test_pts]

    print(f"Patients  — train: {len(train_pts)}  val: {len(val_pts)}  test: {len(test_pts)}")
    print(f"Samples   — train: {len(train)}  val: {len(val)}  test: {len(test)}")
    return train, val, test


if __name__ == "__main__":
    import sys
    split_records(sys.argv[1])
