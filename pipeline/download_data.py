"""Download aggTrades and bookTicker zip files from data.binance.vision."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
import os
import threading

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "default_config.yaml"
_THREAD_LOCAL = threading.local()


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def date_range(start: str, end: str):
    """Yield each date from start to end inclusive."""
    current = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    if current > end_date:
        raise ValueError(f"start date {start} is after end date {end}")
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def _get_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=2)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _THREAD_LOCAL.session = session
    return session


def download_file(url: str, dest: Path) -> int:
    """Download url to dest. Returns bytes downloaded, 0 if 404."""
    resp = _get_session().get(url, stream=True, timeout=30)
    if resp.status_code == 404:
        return 0
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    tmp_dest = dest.with_suffix(dest.suffix + f".part.{os.getpid()}")
    try:
        with open(tmp_dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                f.write(chunk)
                total += len(chunk)
        os.replace(tmp_dest, dest)
        return total
    except Exception:
        if tmp_dest.exists():
            tmp_dest.unlink()
        raise


def _download_task(url: str, dest: Path) -> tuple:
    """Worker function for parallel downloads. Returns (dest_name, bytes, error)."""
    try:
        nbytes = download_file(url, dest)
        return (dest.name, nbytes, None)
    except Exception as e:  # pragma: no cover - network failures are non-deterministic
        return (dest.name, 0, str(e))


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Download Binance futures data")
    parser.add_argument("--start", default=cfg["data"]["start_date"])
    parser.add_argument("--end", default=cfg["data"]["end_date"])
    parser.add_argument("--symbol", default=cfg["data"]["symbol"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist.")
    args = parser.parse_args()

    data_types = ["aggTrades", "bookTicker"]
    base_url = "https://data.binance.vision/data/futures/um/daily"

    all_tasks = []
    skipped_existing = 0
    for current_date in date_range(args.start, args.end):
        for data_type in data_types:
            fname = f"{args.symbol}-{data_type}-{current_date}.zip"
            url = f"{base_url}/{data_type}/{args.symbol}/{fname}"
            dest = BASE_DIR / "data" / "raw" / data_type / fname
            if dest.exists() and not args.force:
                skipped_existing += 1
                continue
            all_tasks.append((url, dest))

    if not all_tasks:
        print(f"Nothing to download. Skipped {skipped_existing} existing files.")
        return

    total_bytes = 0
    errors = []
    missing = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_download_task, url, dest): (url, dest) for url, dest in all_tasks}
        with tqdm(total=len(futures), desc="Downloading") as pbar:
            for future in as_completed(futures):
                name, nbytes, error = future.result()
                if error:
                    errors.append((name, error))
                elif nbytes == 0:
                    missing.append(name)
                else:
                    total_bytes += nbytes
                pbar.update(1)

    print(f"\nQueued downloads: {len(all_tasks)}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Downloaded size: {total_bytes / 1e6:.1f} MB")
    if missing:
        print(f"404 missing files: {len(missing)}")
    if errors:
        print(f"Errors encountered: {len(errors)}")
        for name, err in errors:
            print(f"  {name}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
