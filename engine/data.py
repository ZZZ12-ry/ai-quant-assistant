"""数据加载器：本地研究数据优先，AKShare 连续合约兜底。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import akshare as ak
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent

# 热门品种的主力合约代码
POPULAR_SYMBOLS = {
    "RB": "RB0",
    "HC": "HC0",
    "I": "I0",
    "M": "M0",
    "Y": "Y0",
    "P": "P0",
    "CF": "CF0",
    "TA": "TA0",
    "MA": "MA0",
    "SC": "SC0",
    "AG": "AG0",
    "AU": "AU0",
}

DATA_POLICY = {
    "provider": "AKShare / 新浪财经",
    "api": "futures_main_sina",
    "contract_mode": "主力连续日线",
    "roll_policy": "使用新浪 XX0 主力连续序列，属于已拼接的主力连续行情；平台当前不自行记录换月日期，也未单列换月滑点或展期成本。",
    "research_level": "初筛展示口径，不等同于可实盘验证数据",
}

COLUMN_MAP = {
    "日期": "date",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量": "volume",
    "持仓量": "open_interest",
    "动态结算价": "settlement",
}

ENGLISH_COLUMNS = {"date", "open", "high", "low", "close", "volume", "open_interest"}


def _normalize_ohlcv(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize common Chinese/English OHLCV files to the engine schema."""
    df = df.copy()
    df.rename(columns=COLUMN_MAP, inplace=True)
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    aliases = {
        "datetime": "date",
        "time": "date",
        "trade_date": "date",
        "vol": "volume",
        "oi": "open_interest",
        "openinterest": "open_interest",
        "hold": "open_interest",
    }
    rename = {}
    for alias, canonical in aliases.items():
        if alias in lower_map and canonical not in df.columns:
            rename[lower_map[alias]] = canonical
    for col in list(df.columns):
        low = str(col).strip().lower()
        if low in ENGLISH_COLUMNS and col != low:
            rename[col] = low
    if rename:
        df.rename(columns=rename, inplace=True)

    if "open_interest" not in df.columns:
        df["open_interest"] = 0
    if "volume" not in df.columns:
        df["volume"] = 0

    required = ["date", "open", "high", "low", "close", "volume", "open_interest"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"{source}数据缺少字段: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "open_interest"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[required + [col for col in df.columns if col not in required]]


def _validate(df: pd.DataFrame, source: str) -> pd.DataFrame:
    required = ["date", "open", "high", "low", "close", "volume", "open_interest"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"{source}数据缺少字段: {missing}")
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    if df.empty:
        raise RuntimeError(f"{source}数据为空或关键价格字段全为空")
    df = df.drop_duplicates(subset=["date"], keep="last")
    bad_price = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad_price.any():
        df = df.loc[~bad_price].copy()
    bad_ohlc = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (
        df["low"] > df[["open", "close", "high"]].min(axis=1)
    )
    if bad_ohlc.any():
        df = df.loc[~bad_ohlc].copy()
    if df.empty:
        raise RuntimeError(f"{source}数据价格异常，清洗后为空")
    return df.sort_values("date").reset_index(drop=True)


def _quality_report(df: pd.DataFrame, source: str, symbol: str) -> dict:
    dates = pd.to_datetime(df["date"], errors="coerce").dropna().sort_values()
    gaps = dates.diff().dt.days.dropna()
    return {
        "source": source,
        "symbol": symbol,
        "rows": int(len(df)),
        "start": str(dates.iloc[0].date()) if len(dates) else "",
        "end": str(dates.iloc[-1].date()) if len(dates) else "",
        "duplicate_dates": int(pd.to_datetime(df["date"], errors="coerce").duplicated().sum()),
        "long_calendar_gaps": int((gaps > 10).sum()) if len(gaps) else 0,
        "zero_volume_rows": int((pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0) <= 0).sum()),
    }


def _local_data_candidates(symbol: str) -> list[Path]:
    local = PROJECT_ROOT / "data" / "local"
    return [
        local / f"{symbol}.csv",
        local / f"{symbol}_daily.csv",
        local / f"{symbol}.xlsx",
        local / f"{symbol}_daily.xlsx",
    ]


def _read_local_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    for path in _local_data_candidates(symbol):
        if not path.exists():
            continue
        raw = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
        df = _validate(_normalize_ohlcv(raw, f"本地文件{path.name}"), f"本地文件{path.name}")
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        if df.empty:
            raise RuntimeError(f"本地数据 {path.name} 在请求区间内为空")
        df.attrs["data_policy"] = {
            "provider": "本地研究数据",
            "api": "local_file",
            "contract_mode": "由本地文件决定",
            "roll_policy": "平台不自行推断换月规则；若该文件是移仓换月口径，请保留数据说明。",
            "research_level": "可作为阶段研究主口径，仍需确认数据清洗与换月成本",
            "file": str(path),
            "quality": _quality_report(df, f"本地文件{path.name}", symbol),
            "symbol": symbol,
        }
        print(f"[数据] 使用本地研究数据: {path.name}")
        return df.reset_index(drop=True)
    return None


def describe_data_policy(symbol: str = "", actual: Optional[dict] = None) -> dict:
    """Return the current backtest data policy for report metadata."""
    out = DATA_POLICY.copy()
    if symbol:
        out["symbol"] = symbol
    if actual:
        out.update(actual)
    return out


def get_main_contract_data(
    symbol: str,
    start_date: str = "20200101",
    end_date: str | None = None,
    cache: bool = True,
    prefer_local: bool = True,
) -> pd.DataFrame:
    """获取期货日线数据。

    平台优先读取 `data/local/{symbol}.csv|xlsx` 或
    `data/local/{symbol}_daily.csv|xlsx`。如果没有本地研究数据，则使用
    AKShare/Sina 主力连续日线作为初筛兜底。
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    if prefer_local:
        local_df = _read_local_data(symbol, start_date, end_date)
        if local_df is not None:
            return local_df

    cache_path = PROJECT_ROOT / "data" / f"{symbol}_{start_date}_{end_date}.csv"
    if cache and cache_path.exists():
        df = _validate(_normalize_ohlcv(pd.read_csv(cache_path), f"缓存{cache_path.name}"), f"缓存{cache_path.name}")
        df.attrs["data_policy"] = describe_data_policy(symbol, {
            "provider": "本地缓存(AKShare历史拉取)",
            "file": str(cache_path),
            "quality": _quality_report(df, f"缓存{cache_path.name}", symbol),
        })
        print(f"[数据] 从缓存加载: {cache_path.name}")
        return df

    print(f"[数据] 从AKShare拉取 {symbol} ({start_date}~{end_date})...")
    try:
        raw = ak.futures_main_sina(symbol=symbol, start_date=start_date, end_date=end_date)
    except Exception as exc:
        if cache:
            fallback = sorted(cache_path.parent.glob(f"{symbol}_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            if fallback:
                fp = fallback[0]
                cached = _normalize_ohlcv(pd.read_csv(fp), f"缓存{fp.name}")
                out = _validate(cached, f"缓存{fp.name}")
                out.attrs["data_policy"] = describe_data_policy(symbol, {
                    "provider": "本地缓存(AKShare失败兜底)",
                    "file": str(fp),
                    "quality": _quality_report(out, f"缓存{fp.name}", symbol),
                })
                print(f"[数据] AKShare拉取失败，使用最近缓存: {fp.name}")
                return out
        raise RuntimeError(f"AKShare数据拉取失败，且没有可用缓存: {exc}")

    df = _validate(_normalize_ohlcv(raw, "AKShare"), "AKShare")
    df.attrs["data_policy"] = describe_data_policy(symbol, {"quality": _quality_report(df, "AKShare", symbol)})

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"[数据] 已缓存至: {cache_path}")

    print(f'[数据] 共获取 {len(df)} 条K线 ({df["date"].iloc[0].date()} ~ {df["date"].iloc[-1].date()})')
    return df


def list_available_symbols() -> pd.DataFrame:
    """列出 AKShare 支持的所有期货主力合约。"""
    return ak.futures_display_main_sina()


if __name__ == "__main__":
    frame = get_main_contract_data("RB0", "20240101")
    print(frame.head(3))
    print(frame.tail(3))
