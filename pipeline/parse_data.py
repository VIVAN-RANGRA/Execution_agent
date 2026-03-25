"""Parse raw zip files into Parquet format."""
import argparse
import io
import os
from pathlib import Path
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent

AGG_TRADES_COLS = [
    "agg_trade_id",
    "price",
    "qty",
    "first_trade_id",
    "last_trade_id",
    "timestamp_ms",
    "is_buyer_maker",
]
AGG_TRADES_KEEP = ["timestamp_ms", "price", "qty", "is_buyer_maker"]
AGG_TRADES_DTYPES = {
    "agg_trade_id": "int64",
    "price": "float64",
    "qty": "float64",
    "first_trade_id": "int64",
    "last_trade_id": "int64",
    "timestamp_ms": "int64",
    "is_buyer_maker": "bool",
}

BOOK_TICKER_COLS = [
    "update_id",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "timestamp_ms",
]
BOOK_TICKER_KEEP = ["timestamp_ms", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"]
BOOK_TICKER_DTYPES = {
    "update_id": "int64",
    "best_bid_price": "float64",
    "best_bid_qty": "float64",
    "best_ask_price": "float64",
    "best_ask_qty": "float64",
    "timestamp_ms": "int64",
}

HEADER_TRUE_VALUES = ["true", "True", "TRUE", "1"]
HEADER_FALSE_VALUES = ["false", "False", "FALSE", "0"]
BOOK_TICKER_CHUNK_ROWS = 750_000


def _default_worker_count() -> int:
    """Use a conservative default instead of consuming all CPU cores."""
    cpu_count = os.cpu_count() or 4
    return 6


def _normalize_book_ticker_header_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map Binance bookTicker header variants onto the expected internal schema."""
    normalized = {str(col).strip().lower(): col for col in df.columns}

    def _pick(*aliases: str):
        for alias in aliases:
            source = normalized.get(alias.strip().lower())
            if source is not None:
                return source
        return None

    update_id = _pick("update_id", "u")
    bid_price = _pick("best_bid_price", "b")
    bid_qty = _pick("best_bid_qty", "bq", "B")
    ask_price = _pick("best_ask_price", "a")
    ask_qty = _pick("best_ask_qty", "aq", "A")
    timestamp = _pick("timestamp_ms", "transaction_time", "event_time", "t", "e")

    missing = []
    if update_id is None:
        missing.append("update_id/u")
    if bid_price is None:
        missing.append("best_bid_price/b")
    if bid_qty is None:
        missing.append("best_bid_qty/B")
    if ask_price is None:
        missing.append("best_ask_price/a")
    if ask_qty is None:
        missing.append("best_ask_qty/A")
    if timestamp is None:
        missing.append("timestamp_ms/event_time/transaction_time")
    if missing:
        raise ValueError(f"Missing expected bookTicker header columns: {missing}")

    return pd.DataFrame(
        {
            "update_id": df[update_id],
            "best_bid_price": df[bid_price],
            "best_bid_qty": df[bid_qty],
            "best_ask_price": df[ask_price],
            "best_ask_qty": df[ask_qty],
            "timestamp_ms": df[timestamp],
        }
    )


def _read_csv_payload(
    payload: bytes,
    columns: list[str],
    dtypes: dict[str, str],
    *,
    data_type: str,
) -> pd.DataFrame:
    """Read either headerless or header-present Binance CSV payloads."""
    try:
        return pd.read_csv(
            io.BytesIO(payload),
            header=None,
            names=columns,
            dtype=dtypes,
        )
    except (ValueError, TypeError):
        df = pd.read_csv(io.BytesIO(payload), header=0)
        if data_type == "bookTicker":
            df = _normalize_book_ticker_header_frame(df)
        else:
            if df.shape[1] != len(columns):
                raise ValueError(
                    f"Expected {len(columns)} columns, found {df.shape[1]} in CSV with header"
                )
            df.columns = columns
        return df.astype(dtypes)


def parse_zip(zip_path: Path, data_type: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            payload = f.read()
            if data_type == "aggTrades":
                df = _read_csv_payload(
                    payload,
                    AGG_TRADES_COLS,
                    AGG_TRADES_DTYPES,
                    data_type=data_type,
                )
                df = df[AGG_TRADES_KEEP]
            else:
                df = _read_csv_payload(
                    payload,
                    BOOK_TICKER_COLS,
                    BOOK_TICKER_DTYPES,
                    data_type=data_type,
                )
                df = df[BOOK_TICKER_KEEP]
    assert df.isnull().sum().sum() == 0, f"NaN values found in {zip_path}"
    return df


def _has_header_row(zip_path: Path, csv_name: str) -> bool:
    """Detect whether the CSV inside the zip starts with a header row."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(csv_name) as f:
            first_line = f.readline().decode("utf-8", errors="replace").strip()
    if not first_line:
        return False
    first_token = first_line.split(",", 1)[0].strip().strip('"').strip("'")
    return not first_token.isdigit()


def _coerce_frame(df: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Coerce parsed CSV chunks into the expected schema dtypes."""
    coerced = df.copy()
    for column, dtype in dtypes.items():
        if dtype == "bool":
            series = coerced[column]
            if not pd.api.types.is_bool_dtype(series):
                lowered = series.astype(str).str.strip().str.lower()
                if not lowered.isin({"true", "false", "1", "0"}).all():
                    raise ValueError(f"Unexpected boolean values in column '{column}'")
                coerced[column] = lowered.isin({"true", "1"})
            else:
                coerced[column] = series.astype(bool)
        else:
            coerced[column] = pd.to_numeric(coerced[column], errors="raise").astype(dtype)
    return coerced


def _normalize_chunk(chunk: pd.DataFrame, data_type: str, has_header: bool) -> pd.DataFrame:
    """Normalize a parsed chunk onto the expected internal schema."""
    if data_type == "aggTrades":
        if has_header:
            if chunk.shape[1] != len(AGG_TRADES_COLS):
                raise ValueError(
                    f"Expected {len(AGG_TRADES_COLS)} aggTrades columns, found {chunk.shape[1]}"
                )
            chunk.columns = AGG_TRADES_COLS
        elif chunk.shape[1] != len(AGG_TRADES_COLS):
            raise ValueError(
                f"Expected {len(AGG_TRADES_COLS)} aggTrades columns, found {chunk.shape[1]}"
            )
        return _coerce_frame(chunk[AGG_TRADES_COLS], AGG_TRADES_DTYPES)[AGG_TRADES_KEEP]

    if has_header:
        chunk = _normalize_book_ticker_header_frame(chunk)
    elif chunk.shape[1] != len(BOOK_TICKER_COLS):
        raise ValueError(
            f"Expected {len(BOOK_TICKER_COLS)} bookTicker columns, found {chunk.shape[1]}"
        )
    else:
        chunk.columns = BOOK_TICKER_COLS
    return _coerce_frame(chunk[BOOK_TICKER_COLS], BOOK_TICKER_DTYPES)[BOOK_TICKER_KEEP]


def _iter_normalized_chunks(zip_path: Path, data_type: str):
    """Yield normalized DataFrame chunks for a zipped Binance CSV."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = zf.namelist()[0]
        has_header = _has_header_row(zip_path, csv_name)
        with zf.open(csv_name) as f:
            chunk_iter = pd.read_csv(
                f,
                header=0 if has_header else None,
                names=None if has_header else (AGG_TRADES_COLS if data_type == "aggTrades" else BOOK_TICKER_COLS),
                chunksize=BOOK_TICKER_CHUNK_ROWS if data_type == "bookTicker" else None,
                low_memory=False,
                true_values=HEADER_TRUE_VALUES,
                false_values=HEADER_FALSE_VALUES,
            )
            if isinstance(chunk_iter, pd.DataFrame):
                yield _normalize_chunk(chunk_iter, data_type, has_header)
                return
            for chunk in chunk_iter:
                yield _normalize_chunk(chunk, data_type, has_header)


def _write_parquet_chunks(chunks, out_path: Path) -> None:
    """Write normalized DataFrame chunks into one parquet file atomically."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=out_path.parent,
        suffix=".tmp.parquet",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

    writer = None
    wrote_rows = False
    try:
        for chunk in chunks:
            if chunk.empty:
                continue
            if chunk.isnull().sum().sum() != 0:
                raise ValueError(f"NaN values found while parsing {out_path.name}")
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            writer.write_table(table)
            wrote_rows = True
        if writer is not None:
            writer.close()
            writer = None
        if not wrote_rows:
            raise ValueError(f"No rows parsed for {out_path.name}")
        os.replace(tmp_path, out_path)
    finally:
        if writer is not None:
            writer.close()
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def extract_date_from_filename(fname: str) -> str:
    """BTCUSDT-aggTrades-2024-01-01.zip -> 2024-01-01."""
    return "-".join(fname.replace(".zip", "").split("-")[-3:])


def _parse_single(
    zip_path_str: str,
    data_type: str,
    out_path_str: str,
    overwrite: bool = False,
) -> tuple:
    """Parse a single zip and write parquet. Returns (name, error, skipped)."""
    zip_path = Path(zip_path_str)
    out_path = Path(out_path_str)
    try:
        if out_path.exists() and not overwrite:
            return (zip_path.name, None, True)
        _write_parquet_chunks(_iter_normalized_chunks(zip_path, data_type), out_path)
        return (zip_path.name, None, False)
    except Exception as e:
        return (zip_path.name, str(e), False)


def main():
    parser = argparse.ArgumentParser(description="Parse raw Binance zip files into parquet.")
    parser.add_argument("--workers", type=int, default=_default_worker_count())
    parser.add_argument("--overwrite", action="store_true", help="Rebuild parquet files even if they already exist.")
    args = parser.parse_args()

    for data_type in ["aggTrades", "bookTicker"]:
        raw_dir = BASE_DIR / "data" / "raw" / data_type
        parsed_dir = BASE_DIR / "data" / "parsed" / data_type
        parsed_dir.mkdir(parents=True, exist_ok=True)

        zip_files = sorted(raw_dir.glob("*.zip")) if raw_dir.exists() else []
        if not zip_files:
            print(f"No zip files found in {raw_dir}")
            continue

        tasks = []
        skipped_existing = 0
        for zip_path in zip_files:
            date_str = extract_date_from_filename(zip_path.name)
            out_path = parsed_dir / f"{date_str}.parquet"
            if out_path.exists() and not args.overwrite:
                skipped_existing += 1
                continue
            tasks.append((str(zip_path), data_type, str(out_path), args.overwrite))

        if not tasks:
            print(f"{data_type}: nothing to parse. Skipped {skipped_existing} existing parquet files.")
            continue

        worker_count = 6
        print(
            f"{data_type}: parsing {len(tasks)} files with {worker_count} workers "
            f"(skipped {skipped_existing})."
        )
        errors = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_parse_single, *task): task[0] for task in tasks}
            with tqdm(total=len(futures), desc=f"Parsing {data_type}") as pbar:
                for future in as_completed(futures):
                    name, error, skipped = future.result()
                    if error:
                        errors.append((name, error))
                        tqdm.write(f"  ERROR: {name} - {error}")
                    elif skipped:
                        tqdm.write(f"  SKIP: {name}")
                    pbar.update(1)
        if errors:
            raise SystemExit(f"{data_type}: parsing failed for {len(errors)} file(s)")

    print("Parsing complete.")


if __name__ == "__main__":
    main()
