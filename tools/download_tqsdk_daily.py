"""Download TqSdk daily futures data into data/local for this platform.

This script only uses TqSdk as a data source. It does not use TqSdk's
backtesting framework. The platform will later read the generated CSV through
engine.data.get_main_contract_data(...), because data/local has priority.

Examples:
    python -X utf8 tools/download_tqsdk_daily.py --symbol AU0 --tq-symbol KQ.m@SHFE.au --start 2021-01-01 --end 2026-07-08
    python -X utf8 tools/download_tqsdk_daily.py --symbol RB0 --tq-symbol KQ.m@SHFE.rb --start 2021-01-01 --end 2026-07-08

Credentials:
    set TQ_USER=your_tq_user
    set TQ_PASSWORD=your_tq_password
"""

from __future__ import annotations

import argparse
import os
import tempfile
from contextlib import closing
from datetime import date, datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "local"


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _normalize_downloaded_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    for col in df.columns:
        low = str(col).strip().lower()
        if low == "datetime":
            rename[col] = "date"
        elif low == "open":
            rename[col] = "open"
        elif low == "high":
            rename[col] = "high"
        elif low == "low":
            rename[col] = "low"
        elif low == "close":
            rename[col] = "close"
        elif low == "volume":
            rename[col] = "volume"
        elif low in {"open_oi", "close_oi", "open_interest"}:
            rename[col] = "open_interest"
    df = df.rename(columns=rename)
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"TqSdk downloaded CSV missing fields: {missing}; columns={list(df.columns)}")
    if "open_interest" not in df.columns:
        df["open_interest"] = 0
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    for col in ["open", "high", "low", "close", "volume", "open_interest"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df[["date", "open", "high", "low", "close", "volume", "open_interest"]]
    return df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)


def download_daily(symbol: str, tq_symbol: str, start: date, end: date, out_file: Path) -> Path:
    try:
        from tqsdk import TqApi, TqAuth
        from tqsdk.tools import DataDownloader
    except ImportError as exc:
        raise RuntimeError("请先安装 TqSdk：pip install tqsdk") from exc

    user = os.environ.get("TQ_USER", "").strip()
    password = os.environ.get("TQ_PASSWORD", "").strip()
    if not user or not password:
        raise RuntimeError("请先设置环境变量 TQ_USER 和 TQ_PASSWORD，不要把账号密码写进代码。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw_csv = Path(tmp) / f"{symbol}_raw.csv"
        api = TqApi(auth=TqAuth(user, password))
        task = DataDownloader(
            api,
            symbol_list=tq_symbol,
            dur_sec=24 * 60 * 60,
            start_dt=start,
            end_dt=end,
            csv_file_name=str(raw_csv),
        )
        with closing(api):
            last_progress = -1
            while not task.is_finished():
                api.wait_update()
                progress = int(task.get_progress())
                if progress != last_progress:
                    print(f"[TqSdk] {tq_symbol} download {progress}%")
                    last_progress = progress
        df = _normalize_downloaded_csv(raw_csv)
    if df.empty:
        raise RuntimeError("下载完成但清洗后为空，请检查合约代码和日期范围。")
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TqSdk daily data to data/local.")
    parser.add_argument("--symbol", required=True, help="Platform symbol, e.g. AU0, RB0.")
    parser.add_argument("--tq-symbol", required=True, help="TqSdk symbol, e.g. KQ.m@SHFE.au.")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output", default="", help="Optional output CSV path.")
    args = parser.parse_args()

    symbol = args.symbol.strip().upper()
    out_file = Path(args.output) if args.output else OUT_DIR / f"{symbol}.csv"
    path = download_daily(symbol, args.tq_symbol.strip(), _parse_date(args.start), _parse_date(args.end), out_file)
    print(f"[OK] saved: {path}")


if __name__ == "__main__":
    main()
