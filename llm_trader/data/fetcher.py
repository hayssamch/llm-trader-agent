"""
data/fetcher.py — UC: Acquire market data
Downloads OHLCV via yfinance with local Parquet cache, OR generates
synthetic OHLCV when config.use_synthetic is True (offline smoke testing).

Leakage guard: fetch_window(as_of) returns only bars strictly before as_of.
"""
import os
import logging
from typing import Optional

import numpy as np
import pandas as pd

from llm_trader.config import AgentConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self, config: AgentConfig = DEFAULT_CONFIG):
        self.config = config
        os.makedirs(config.cache_dir, exist_ok=True)
        # Where the prices of the most recent fetch() came from: "yfinance",
        # "alphavantage", "cache" or "synthetic". Yahoo (adjusted prices) and Alpha Vantage
        # (unadjusted) can differ slightly, so the live log records which one was used.
        self.last_source = "unknown"

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def fetch(
        self,
        ticker:   str,
        start:    Optional[str] = None,
        end:      Optional[str] = None,
        interval: Optional[str] = None,
        force:    bool = False,
    ) -> pd.DataFrame:
        start    = start    or self.config.start_date
        end      = end      or self.config.end_date
        interval = interval or self.config.interval

        if self.config.use_synthetic:
            self.last_source = "synthetic"
            return self._synthetic(ticker, start, end)

        cache_path = self._cache_path(ticker, start, end, interval)
        if not force and os.path.exists(cache_path):
            logger.debug("Cache hit: %s", cache_path)
            self.last_source = "cache"
            return self._ensure_utc(pd.read_parquet(cache_path))

        logger.info("Downloading %s (%s → %s, %s)", ticker, start, end, interval)
        df = self._download(ticker, start, end, interval)
        df.to_parquet(cache_path)
        return df

    def fetch_window(self, ticker: str, as_of: str, n_bars: int = 252) -> pd.DataFrame:
        """Last n_bars trading days STRICTLY BEFORE as_of (leakage-safe)."""
        full   = self.fetch(ticker)
        cutoff = pd.Timestamp(as_of, tz="UTC")
        past   = full[full.index < cutoff]
        return past.iloc[-n_bars:].copy()

    def trading_dates(self, ticker: str) -> pd.DatetimeIndex:
        return self.fetch(ticker).index

    # ─────────────────────────────────────────────────────────────────────────
    # Download: yfinance first, Alpha Vantage fallback. Yahoo rate-limits
    # cloud/datacenter IPs (e.g. Streamlit Cloud → YFRateLimitError); Alpha
    # Vantage works there with a free API key (env var ALPHAVANTAGE_API_KEY).
    # ─────────────────────────────────────────────────────────────────────────

    def _download(self, ticker, start, end, interval) -> pd.DataFrame:
        try:
            df = self._download_yfinance(ticker, start, end, interval)
            self.last_source = "yfinance"
            return df
        except Exception as exc:
            av_key = os.environ.get("ALPHAVANTAGE_API_KEY")
            if av_key and interval == "1d":
                logger.warning("yfinance failed for %s (%s) — trying Alpha Vantage",
                               ticker, exc)
                df = self._download_alphavantage(ticker, start, end, av_key)
                self.last_source = "alphavantage"
                return df
            raise ValueError(
                f"No data for {ticker} ({start} → {end}). yfinance failed ({exc}). "
                f"On cloud hosts Yahoo often rate-limits; set ALPHAVANTAGE_API_KEY "
                f"(free key from alphavantage.co) to enable the fallback source."
            ) from exc

    def _download_yfinance(self, ticker, start, end, interval) -> pd.DataFrame:
        import yfinance as yf
        raw = yf.download(ticker, start=start, end=end, interval=interval,
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"No data for {ticker} ({start} → {end})")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
        return self._ensure_utc(df)

    def _download_alphavantage(self, ticker, start, end, api_key) -> pd.DataFrame:
        # TIME_SERIES_DAILY is free-tier (25 req/day) and unadjusted; adequate
        # for recent windows without splits. Results are cached to parquet.
        url = ("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
               f"&symbol={ticker}&outputsize=full&datatype=csv&apikey={api_key}")
        raw = pd.read_csv(url)
        if "timestamp" not in raw.columns:
            detail = raw.iloc[0, 0] if len(raw) else "empty response"
            raise ValueError(f"Alpha Vantage error for {ticker}: {detail}")
        raw["timestamp"] = pd.to_datetime(raw["timestamp"])
        raw = raw.set_index("timestamp").sort_index()
        raw = raw.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                  "close": "Close", "volume": "Volume"})
        df = raw.loc[start:end, ["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
        if df.empty:
            raise ValueError(f"No data for {ticker} ({start} → {end})")
        logger.info("Alpha Vantage fallback data: %s (%d bars)", ticker, len(df))
        return self._ensure_utc(df)

    # ─────────────────────────────────────────────────────────────────────────
    # Synthetic generator (offline testing) — geometric brownian motion + noise
    # ─────────────────────────────────────────────────────────────────────────

    def _synthetic(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Deterministic per-ticker synthetic OHLCV so smoke tests are reproducible.
        Not for research use — only to verify pipeline wiring without network.
        """
        seed = abs(hash(ticker)) % (2**32)
        rng  = np.random.default_rng(seed)

        dates = pd.bdate_range(start=start, end=end, tz="UTC")
        n     = len(dates)
        if n == 0:
            raise ValueError(f"Empty synthetic date range {start} → {end}")

        mu, sigma = 0.0004, 0.018            # daily drift & vol
        rets   = rng.normal(mu, sigma, n)
        price0 = 100 + (seed % 200)          # ticker-dependent start price
        close  = price0 * np.cumprod(1 + rets)

        intraday = np.abs(rng.normal(0, sigma / 2, n))
        high  = close * (1 + intraday)
        low   = close * (1 - intraday)
        open_ = np.concatenate([[close[0]], close[:-1]])
        vol   = rng.integers(1_000_000, 20_000_000, n)

        df = pd.DataFrame(
            {"Open": open_, "High": np.maximum(high, open_),
             "Low": np.minimum(low, open_), "Close": close, "Volume": vol},
            index=dates,
        )
        logger.info("Synthetic data generated: %s (%d bars)", ticker, n)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df

    def _cache_path(self, ticker, start, end, interval) -> str:
        key = f"{ticker}_{start}_{end}_{interval}".replace(" ", "_")
        return os.path.join(self.config.cache_dir, f"{key}.parquet")
