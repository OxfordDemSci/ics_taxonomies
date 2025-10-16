# ics_grants.py
# -*- coding: utf-8 -*-
"""
Scrape REF 'Grant funding' tables, return row-by-row and aggregated DataFrames.

Public API
----------
get_ics_grants(
    input_csv_path="../data/complete/enhanced_ref_data.csv",
    base_url="https://results2021.ref.ac.uk/impact",
    out_rows_csv="../data/ics_grants/ICS_grants_row_by_row.csv",
    out_agg_csv="../data/ics_grants/ICS_grants_aggregated.csv",
    request_timeout=30,
    user_agent=("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    show_progress=True,
)

Returns
-------
grants_df : pd.DataFrame
    Columns: ["ics_id","source_url","grant_number","value_raw","currency","value_numeric"]
grants_collapsed : pd.DataFrame
    Columns:
      ["REF impact case study identifier",
       "grant_number", "value_raw", "currency", "total_value_numeric"]
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Iterable, Iterator, Tuple, Optional

import requests
import pandas as pd

# Progress bar is optional at runtime.
try:  # pragma: no cover
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

# ---------------------------
# HTML parsing (robust setup)
# ---------------------------
try:
    from bs4 import BeautifulSoup
    _BS_PARSER = "lxml"
except Exception:  # Fall back to stdlib html.parser
    from bs4 import BeautifulSoup  # type: ignore
    _BS_PARSER = "html.parser"

# ---------------------------
# Currency parsing utilities
# ---------------------------
_CURRENCY_RE = re.compile(r"^\s*([^\d\s\-+])?\s*([-+]?\d[\d,]*(?:\.\d+)?)")

def _parse_currency(value_str: str) -> Tuple[Optional[str], Optional[Decimal]]:
    """
    Parse a human-formatted currency string.

    Parameters
    ----------
    value_str : str
        Example: "£10,000,000" or "€ 1,234.56"

    Returns
    -------
    (symbol, value) : (Optional[str], Optional[Decimal])
        `value` is in whole currency units as Decimal; returns (None, None) on failure.
    """
    if not value_str:
        return None, None
    s = value_str.replace("\xa0", " ").strip()
    m = _CURRENCY_RE.match(s)
    if not m:
        return None, None
    symbol, num = m.groups()
    try:
        num_clean = num.replace(",", "")
        return symbol, Decimal(num_clean)
    except (InvalidOperation, ValueError):
        return symbol, None

# ---------------------------
# HTML table extraction
# ---------------------------
def _extract_grants_from_html(html: str, url: str) -> Iterator[dict]:
    """
    Yield rows from the 'Grant funding' table.

    Expected headers in <thead>: 'Grant number', 'Value of grant'.
    Strategy:
      A) Find heading 'Grant funding' and then the first table within its container; else
      B) First table whose thead contains the required headers.
    """
    soup = BeautifulSoup(html, _BS_PARSER)

    # Strategy A
    header = soup.find(
        lambda tag: tag.name in ("h2", "h3", "h4")
        and tag.get_text(strip=True).lower() == "grant funding"
    )
    table = None
    if header is not None:
        container = header.find_parent(["div", "section", "article"]) or header.parent
        if container:
            table = container.find("table")

    # Strategy B
    if table is None:
        def _thead_matches(t):
            thead = t.find("thead")
            if thead is None:
                return False
            headers = [th.get_text(strip=True).lower() for th in thead.find_all("th")]
            return ("grant number" in headers) and ("value of grant" in headers)
        for t in soup.find_all("table"):
            if _thead_matches(t):
                table = t
                break

    if table is None:
        return  # nothing to yield

    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        grant_number = tds[0].get_text(separator=" ", strip=True)
        value_raw = tds[1].get_text(separator=" ", strip=True)
        currency, value_numeric = _parse_currency(value_raw)
        yield {
            "source_url": url,
            "grant_number": grant_number,
            "value_raw": value_raw,
            "currency": currency,
            "value_numeric": value_numeric,
        }

# ---------------------------
# Public pipeline function
# ---------------------------
def get_ics_grants(
    input_csv_path: str = "../data/complete/enhanced_ref_data.csv",
    base_url: str = "https://results2021.ref.ac.uk/impact",
    out_rows_csv: Optional[str] = "../data/ics_grants/ICS_grants_row_by_row.csv",
    out_agg_csv: Optional[str] = "../data/ics_grants/ICS_grants_aggregated.csv",
    request_timeout: int = 30,
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scrape all ICS pages listed in `input_csv_path`, extract Grant funding rows,
    and return both the row-by-row and aggregated DataFrames.

    The input CSV must contain a column:
        'REF impact case study identifier'
    """
    ids = (
        pd.read_csv(input_csv_path)["REF impact case study identifier"]
        .astype(str)
        .tolist()
    )

    grants_df = pd.DataFrame(
        columns=["ics_id", "source_url", "grant_number", "value_raw", "currency", "value_numeric"]
    )

    headers = {"User-Agent": user_agent}
    iterator: Iterable[str] = ids
    if show_progress:
        iterator = tqdm(ids, desc="Scraping 'Grant funding' rows")

    with requests.Session() as session:
        session.headers.update(headers)
        for ics_id in iterator:
            url = f"{base_url}/{ics_id}"
            try:
                r = session.get(url, timeout=request_timeout)
                r.raise_for_status()
            except Exception:
                continue

            for row in _extract_grants_from_html(r.text, url=url):
                row["ics_id"] = ics_id
                grants_df.loc[len(grants_df)] = row

    # Row-by-row CSV
    if out_rows_csv:
        _ensure_parent_dir(out_rows_csv)
        grants_df.to_csv(out_rows_csv, index=False)

    # Aggregation
    grants_df_named = grants_df.rename(columns={"ics_id": "REF impact case study identifier"})
    grants_collapsed = (
        grants_df_named
        .groupby("REF impact case study identifier", as_index=False)
        .agg({
            "grant_number": lambda x: list(x.dropna().astype(str)),
            "value_raw":    lambda x: list(x.dropna().astype(str)),
            "currency":     lambda x: list(x.dropna().astype(str)),
            "value_numeric": "sum",
        })
        .rename(columns={"value_numeric": "total_value_numeric"})
    )
    grants_collapsed["total_value_numeric"] = pd.to_numeric(
        grants_collapsed["total_value_numeric"], errors="coerce"
    ).fillna(0)

    if out_agg_csv:
        _ensure_parent_dir(out_agg_csv)
        grants_collapsed.to_csv(out_agg_csv, index=False)

    return grants_df, grants_collapsed

# ---------------------------
# Utilities
# ---------------------------
def _ensure_parent_dir(path: str) -> None:
    import os
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    rows, agg = get_ics_grants()
    print(f"Collected {len(rows)} rows from {rows['ics_id'].nunique()} case studies.")
    print(agg.head())
