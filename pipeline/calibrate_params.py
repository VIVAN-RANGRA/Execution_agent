"""Calibrate Almgren-Chriss parameters from historical data."""
from pathlib import Path
import json
import os
import tempfile
import pandas as pd
import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "default_config.yaml"

def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    train_split = cfg["data"]["train_split"]
    eta_multiplier = cfg["impact_model"]["eta_multiplier"]
    gamma_multiplier = cfg["impact_model"]["gamma_multiplier"]
    alpha = cfg["impact_model"]["alpha"]

    parsed_dir = BASE_DIR / "data" / "parsed" / "aggTrades"
    parquet_files = sorted(parsed_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {parsed_dir}")

    n_train = max(1, int(len(parquet_files) * train_split))
    train_files = parquet_files[:n_train]
    print(f"Calibrating on {n_train} training days...")

    daily_sigmas = []
    daily_advs = []

    for f in train_files:
        df = pd.read_parquet(f, columns=["timestamp_ms", "price", "qty"])
        # Resample to 1-second OHLC: group by second, take last price
        df["second"] = df["timestamp_ms"] // 1000
        ohlc = df.groupby("second")["price"].last().reset_index()
        ohlc = ohlc.sort_values("second")
        if len(ohlc) < 2:
            continue
        log_returns = np.diff(np.log(ohlc["price"].values))
        sigma = float(np.std(log_returns))
        adv = float(df["qty"].sum())
        if not np.isfinite(sigma) or not np.isfinite(adv) or adv <= 0:
            continue
        daily_sigmas.append(sigma)
        daily_advs.append(adv)

    if not daily_sigmas:
        raise ValueError("No valid days to calibrate from")

    mean_sigma = float(np.mean(daily_sigmas))
    mean_adv = float(np.mean(daily_advs))
    if mean_adv <= 0 or not np.isfinite(mean_adv):
        raise ValueError("Average daily volume is non-positive; cannot calibrate impact parameters")
    if not np.isfinite(mean_sigma):
        raise ValueError("Average sigma is non-finite; cannot calibrate impact parameters")
    eta = eta_multiplier * mean_sigma * np.sqrt(1.0 / mean_adv)
    gamma = gamma_multiplier * eta

    date_range_str = f"{train_files[0].stem} to {train_files[-1].stem}"

    params = {
        "sigma_per_second": mean_sigma,
        "adv_btc": mean_adv,
        "eta": eta,
        "gamma": gamma,
        "alpha": alpha,
        "calibration_date_range": date_range_str
    }

    out_path = BASE_DIR / "config" / "calibration_params.json"
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=out_path.parent,
        suffix=".tmp.json",
        mode="w",
    ) as tmp_file:
        json.dump(params, tmp_file, indent=2)
        tmp_path = Path(tmp_file.name)
    try:
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    print(f"Calibration parameters saved to {out_path}")
    print(f"  sigma_per_second: {mean_sigma:.8f}")
    print(f"  adv_btc:          {mean_adv:.4f}")
    print(f"  eta:              {eta:.8f}")
    print(f"  gamma:            {gamma:.8f}")
    print(f"  alpha:            {alpha}")
    print(f"  date_range:       {date_range_str}")
    print("Next step:          python pipeline/build_feature_stats.py")

if __name__ == "__main__":
    main()
