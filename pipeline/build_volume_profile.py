"""Build intraday volume profile from aggTrades training data."""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "default_config.yaml"


def _minute_volume_sum(parquet_path: str) -> pd.Series:
    df = pd.read_parquet(parquet_path, columns=["timestamp_ms", "qty"])
    minutes = ((df["timestamp_ms"].to_numpy(dtype=np.int64) // 1000 // 60) % 1440).astype(np.int64)
    qty = df["qty"].to_numpy(dtype=np.float64)
    volume_by_minute = np.bincount(minutes, weights=qty, minlength=1440).astype(np.float64)
    return pd.Series(volume_by_minute)


def main():
    parser = argparse.ArgumentParser(description="Build 1440-bucket intraday volume profile.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    train_split = cfg["data"]["train_split"]
    parsed_dir = BASE_DIR / "data" / "parsed" / "aggTrades"
    parquet_files = sorted(parsed_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parsed_dir}")

    n_train = max(1, int(len(parquet_files) * train_split))
    train_files = parquet_files[:n_train]
    print(f"Using {n_train}/{len(parquet_files)} days for volume profile")

    minute_totals = np.zeros(1440, dtype=np.float64)
    max_workers = min(args.workers, len(train_files))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_minute_volume_sum, str(path)): path for path in train_files}
        for future in as_completed(futures):
            minute_totals += future.result().to_numpy(dtype=np.float64, copy=False)

    profile = pd.DataFrame({
        "minute_of_day": np.arange(1440, dtype=np.int64),
        "fraction": minute_totals,
    })
    total_volume = float(profile["fraction"].sum())
    if total_volume <= 0:
        raise ValueError("Training data has zero aggregate volume; cannot build volume profile")
    profile["fraction"] = profile["fraction"] / total_volume
    assert abs(profile["fraction"].sum() - 1.0) < 1e-10, "Fractions do not sum to 1.0"

    out_path = BASE_DIR / "data" / "volume_profile.parquet"
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=out_path.parent,
        suffix=".tmp.parquet",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        profile.to_parquet(tmp_path, engine="pyarrow", index=False)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    print(f"Volume profile saved to {out_path}")
    print(f"Profile shape: {profile.shape}, fraction sum: {profile['fraction'].sum():.10f}")


if __name__ == "__main__":
    main()
