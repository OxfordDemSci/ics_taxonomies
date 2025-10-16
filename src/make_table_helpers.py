from __future__ import annotations

import os, zipfile, glob
from typing import Dict, Iterable, Optional, List
import numpy as np
import pandas as pd


def build_theme_table(
    zip_path: str,
    extract_dir: Optional[str] = None,
    output_csv: Optional[str] = None,
    column_aliases: Optional[Dict[str, Iterable[str]]] = None,
) -> pd.DataFrame:
    """
    Build the grouped table from a ZIP containing a CSV/XLS(X) file.

    Parameters
    ----------
    zip_path : str
        Path to the .zip archive containing the dataset (CSV or Excel).
    extract_dir : Optional[str]
        Where to extract the archive. If None, uses "<zip_path>__extracted".
    output_csv : Optional[str]
        If provided, the resulting table is also written here as CSV.
    column_aliases : Optional[Dict[str, Iterable[str]]]
        Optional mapping to override/extend candidate column names.
        Keys: "theme", "topic1", "uoa", "panel", "value", "male", "female",
              "institution", "redbrick".
        Values: iterable of candidate strings (case/spacing/punctuation ignored).

    Returns
    -------
    pd.DataFrame
        A table with columns:
        ["Theme Name","Number of ICS","Modal Topic","Modal UoA","Modal Panel",
         "Average Grant","Percent Redbrick","M/F ratio"] sorted by Theme Name.

    Notes
    -----
    - Column names are normalised: lowercase, non-alnum → underscore, and duplicates removed.
    - Mode is defined as any maximiser of the frequency function; ties are broken
      by first appearance in the group (deterministic).
    - "Average Grant" treats NaNs as 0 **for the averaging step only**, as requested.
    - "Percent Redbrick" is computed over unique institutions (if available),
      otherwise row-level mean; Boolean truth set: {1,true,t,yes,y,redbrick}.
    - "M/F ratio" = (#male) / (#female) at the group level; division by zero → NaN.
    """
    # ---------- 0) Column alias defaults ----------
    defaults: Dict[str, List[str]] = {
        # The user’s snippet uses "topic1_theme_long"; include common variants too
        "theme": [
            "topic1_theme_long", "topic1_theme", "topic_1_theme", "topic1theme", "theme"
        ],
        "topic1": ["topic1_name", "topic_1", "topic1", "topic"],
        "uoa": ["unit_of_assessment", "uoa"],
        "panel": ["main_panel", "panel"],
        "value": ["total_value_numeric", "total_value", "grant_amount", "total_amount"],
        "male": ["number_male", "male", "n_male"],
        "female": ["number_female", "female", "n_female"],
        "institution": ["institution", "institution_name", "university", "organisation", "organization"],
        "redbrick": ["redbrick", "redbrick_flag", "is_redbrick"],
    }
    if column_aliases:
        # Prepend user-supplied aliases to take precedence
        for k, v in column_aliases.items():
            defaults[k] = list(v) + defaults.get(k, [])

    # ---------- 1) Extract and load ----------
    if extract_dir is None:
        extract_dir = f"{os.path.splitext(zip_path)[0]}__extracted"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    candidates = (
        glob.glob(os.path.join(extract_dir, "**", "*.csv"), recursive=True)
        + glob.glob(os.path.join(extract_dir, "**", "*.xlsx"), recursive=True)
        + glob.glob(os.path.join(extract_dir, "**", "*.xls"), recursive=True)
    )
    if not candidates:
        raise FileNotFoundError("No CSV/Excel file found inside the ZIP.")

    data_path = candidates[0]
    if data_path.lower().endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)

    # ---------- 2) Normalise names, then DEDUPLICATE ----------
    def _norm(s: str) -> str:
        return pd.Series([s]).str.strip().str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).iloc[0]

    df.columns = [_norm(str(c)) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # ---------- 3) Column selection ----------
    def _pick(cols: Iterable[str]) -> str:
        for c in cols:
            c_norm = _norm(c)
            if c_norm in df.columns:
                return c_norm
        raise KeyError(f"Missing required column; tried {list(cols)}")

    c_theme  = _pick(defaults["theme"])
    c_topic1 = _pick(defaults["topic1"])
    c_uoa    = _pick(defaults["uoa"])
    c_panel  = _pick(defaults["panel"])

    def _first_present(keys: Iterable[str]) -> Optional[str]:
        for c in keys:
            c_norm = _norm(c)
            if c_norm in df.columns:
                return c_norm
        return None

    c_value  = _first_present(defaults["value"])
    c_male   = _first_present(defaults["male"])
    c_female = _first_present(defaults["female"])
    c_inst   = _first_present(defaults["institution"])
    c_red    = _first_present(defaults["redbrick"])

    # ---------- 4) Coerce numerics ----------
    if c_value is None:
        df["__value__"] = 0.0
        c_value = "__value__"
    df[c_value] = pd.to_numeric(df[c_value], errors="coerce").fillna(0)

    # male/female can remain float with NaNs for correct summation semantics
    if c_male is not None:
        df[c_male] = pd.to_numeric(df[c_male], errors="coerce")
    if c_female is not None:
        df[c_female] = pd.to_numeric(df[c_female], errors="coerce")

    # ---------- 5) Aggregation ----------
    g = df.groupby(c_theme, dropna=False)

    # Deterministic mode: break ties by first appearance within group
    def _mode_first(s: pd.Series):
        s = s.dropna()
        if s.empty:
            return np.nan
        counts = s.value_counts(dropna=False)
        max_count = counts.max()
        # pick the first value in original order among those achieving max_count
        winners = set(counts[counts == max_count].index.tolist())
        for v in s:
            if v in winners:
                return v
        return np.nan  # unreachable

    out = pd.DataFrame({
        "Number of ICS": g.size(),
        "Modal Topic":   g[c_topic1].apply(_mode_first),
        "Modal UoA":     g[c_uoa].apply(_mode_first),
        "Modal Panel":   g[c_panel].apply(_mode_first),
        "Average Grant": g[c_value].mean(),
    })

    # Percent Redbrick
    if c_red is not None:
        rb = df[c_red].astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y", "redbrick"})
        df["_is_red_"] = rb
        if c_inst is not None:
            inst_rb = (
                df.groupby([c_theme, df[c_inst]])["_is_red_"]
                  .max()                      # institution considered redbrick if any record says so
                  .groupby(level=0).mean()*100
            )
            out["Percent Redbrick"] = inst_rb
        else:
            out["Percent Redbrick"] = g["_is_red_"].mean()*100
    else:
        out["Percent Redbrick"] = np.nan

    # M/F ratio
    if c_male is not None and c_female is not None:
        male_sum   = g[c_male].sum(min_count=1)
        female_sum = g[c_female].sum(min_count=1).replace(0, np.nan)
        out["M/F ratio"] = male_sum / female_sum
    else:
        out["M/F ratio"] = np.nan

    # ---------- 6) Final ordering ----------
    out = (
        out.reset_index()
           .rename(columns={c_theme: "Theme Name"})
           .loc[:, ["Theme Name","Number of ICS","Modal Topic","Modal UoA","Modal Panel",
                    "Average Grant","Percent Redbrick","M/F ratio"]]
           .sort_values("Theme Name", kind="stable")
           .reset_index(drop=True)
    )

    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        out.to_csv(output_csv, index=False)

    return out
