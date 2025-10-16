# ref_staff.py
# -*- coding: utf-8 -*-
"""
REF staff extraction pipeline with always-on OpenAI integration.

Reads the API key from:  ../keys/OPENAI_API_KEY
Requires:
    - pandas, requests, tqdm, pdfminer.six
    - openai
    - gender-guesser
"""

from __future__ import annotations
import io, os, re, json, time, logging, requests, pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from openai import OpenAI

# =========================
# 0) OPENAI INITIALISATION
# =========================

def _load_openai_client() -> OpenAI:
    key_path = os.path.join(os.path.dirname(__file__), "../keys/OPENAI_API_KEY")
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"OpenAI key not found at {key_path}")
    with open(key_path, "r", encoding="utf-8") as f:
        key = f.read().strip()
    if not key.startswith("sk-") and not key.startswith("proj-") and not key.startswith("sk-proj-"):
        raise ValueError("The file ../keys/OPENAI_API_KEY does not contain a valid OpenAI API key.")
    return OpenAI(api_key=key)

client = _load_openai_client()

# =========================
# 1) PDF TEXT EXTRACTION
# =========================

def _extract_text_pdfminer(pdf_bytes: bytes) -> str:
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    from pdfminer.high_level import extract_text
    buf = io.BytesIO(pdf_bytes)
    import contextlib
    sink = io.StringIO()
    with contextlib.redirect_stderr(sink):
        try:
            t = extract_text(buf) or ""
        except Exception:
            t = ""
    return t.strip()

def _extract_text_pymupdf(pdf_bytes: bytes) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        text = "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()
    return (text or "").strip()

def extract_text_safe(pdf_bytes: bytes) -> str:
    try:
        return _extract_text_pymupdf(pdf_bytes)
    except Exception:
        return _extract_text_pdfminer(pdf_bytes)

# =========================
# 2) STAFF BLOCK ISOLATION
# =========================

DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212"
RE_DASHES      = re.compile(f"[{DASHES}]")
RE_PAGE        = re.compile(r"\n?Page\s+\d+\s*\n", flags=re.I)
RE_MULTI_SPACE = re.compile(r"[ \t]+")

def _norm_text(s: str) -> str:
    if not isinstance(s, str) or not s.strip(): return ""
    s = s.replace("\r", "")
    s = RE_DASHES.sub("-", s)
    s = RE_PAGE.sub("\n", s)
    return re.sub(r"\n{3,}", "\n\n", s)

# Original specific patterns kept (but not relied upon for start detection)
PAT_NAMES   = re.compile(r"Name\(s\)\s*:", flags=re.I)
PAT_ROLES   = re.compile(r"Role\(s\)\s*(?:\(\s*e\.?g\.?\s*job\s*title\s*\))?\s*:", flags=re.I)
PAT_PERIODS = re.compile(r"Period\(s\)\s*employed\s*by\s*[\s\S]*?submitting\s*HEI\s*:", flags=re.I)

# Generalised start-of-line headings: match 'name*:', 'role*:', 'period*:'
PAT_START_NAME   = re.compile(r"^\s*name.*?:", flags=re.I | re.M)
PAT_START_ROLE   = re.compile(r"^\s*role.*?:", flags=re.I | re.M)
PAT_START_PERIOD = re.compile(r"^\s*period.*?:", flags=re.I | re.M)

NEXT_SECTION_MARKERS = [
    re.compile(r"^\s*Period\s*when\s*the\s*claimed\s*impact\s*occurred\s*:", re.I | re.M),
    re.compile(r"^\s*\d+\.\s*Summary\s*of\s*the\s*impact", re.I | re.M),
    re.compile(r"^\s*\d+\.\s*Underpinning\s*research", re.I | re.M),
    re.compile(r"^\s*\d+\.\s*References\s*to\s*the\s*research", re.I | re.M),
    re.compile(r"^\s*\d+\.\s*Details\s*of\s*the\s*impact", re.I | re.M),
]

def isolate_staff_names_block(text: str) -> Optional[str]:
    """
    Generalised block extraction:
    Start the block at the earliest of a line beginning with 'name*:', 'role*:', or 'period*:' (case-insensitive),
    allowing arbitrary characters between the keyword and the colon. Normalise headings within the block to:
      - 'Name(s):'
      - 'Role(s):'
      - 'Period(s) employed by submitting HEI:'
    """
    txt = _norm_text(text)
    if not txt:
        return None

    # Find earliest start among the three headings
    starts = []
    for pat in (PAT_START_NAME, PAT_START_ROLE, PAT_START_PERIOD):
        m = pat.search(txt)
        if m:
            starts.append(m.start())
    if not starts:
        return None
    start = min(starts)

    # Determine end using next-section markers
    next_hits = [pat.search(txt, pos=start) for pat in NEXT_SECTION_MARKERS]
    ends = [m.start() for m in next_hits if m]
    end = min(ends) if ends else len(txt)

    block = txt[start:end].strip()
    if not block:
        return None

    # Normalise headings to canonical labels, only when they begin a line
    def _sub_heading(pattern: re.Pattern, replacement: str, s: str) -> str:
        return re.sub(pattern, replacement, s)

    block = _sub_heading(PAT_START_NAME,   "Name(s):", block)
    block = _sub_heading(PAT_START_ROLE,   "Role(s):", block)
    block = _sub_heading(PAT_START_PERIOD, "Period(s) employed by submitting HEI:", block)

    # Also normalise any legacy specific patterns to the canonical labels (idempotent)
    block = PAT_NAMES.sub("Name(s):", block)
    block = PAT_ROLES.sub("Role(s):", block)
    block = PAT_PERIODS.sub("Period(s) employed by submitting HEI:", block)

    paras = [RE_MULTI_SPACE.sub(" ", " ".join(p.strip() for p in para.splitlines())).strip()
             for para in re.split(r"(?:\n\s*){2,}", block)]
    block = "\n\n".join(p for p in paras if p)
    return block or None

# =========================
# 3) NAME NORMALISATION
# =========================

TITLE_PREFIXES = [r"professor", r"prof", r"dr", r"sir", r"dame", r"mr", r"mrs", r"ms", r"miss"]
TITLE_SUFFIXES = [r"phd", r"dphil", r"md", r"frs", r"frse", r"freng", r"obe", r"cbe", r"mbe"]

RE_TITLE_PREFIX = re.compile(rf"^({'|'.join(TITLE_PREFIXES)})\b\.?\s+", flags=re.I)
RE_TITLE_SUFFIX = re.compile(rf"\b,?\s+({'|'.join(TITLE_SUFFIXES)})\.?\b\.?", flags=re.I)

def strip_titles(name: str) -> str:
    if not isinstance(name, str): return name
    n = name.strip()
    changed = True
    while changed:
        changed = False
        n2 = RE_TITLE_PREFIX.sub("", n)
        n2 = RE_TITLE_SUFFIX.sub("", n2).strip(" ,")
        if n2 != n:
            n = n2
            changed = True
    return n

def normalize_name(name: str) -> str:
    if not isinstance(name, str): return name
    t = name.strip()
    if not t: return t
    tokens = [w[:1].upper() + w[1:] if len(w) > 1 else w.upper() for w in t.split()]
    return " ".join(tokens)

def extract_given_name(name_no_titles: str) -> str:
    if not isinstance(name_no_titles, str) or not name_no_titles.strip():
        return ""
    toks = re.split(r"[ \-]+", name_no_titles.strip())
    for t in toks:
        if re.fullmatch(r"[A-Za-z]\.?([A-Za-z]\.)?", t): continue
        if t.lower() in {"van", "von", "de", "del", "du", "da"}: continue
        return t
    return toks[0] if toks else ""

# =========================
# 4) LLM PARSING
# =========================

_SYSTEM_MSG = (
    "You are given REF 'Details of staff' blocks starting at 'Name(s):' and possibly including "
    "'Role(s):'. The sections are PARALLEL LISTS. "
    "Return JSON {'people': [{'name': ..., 'roles': [...]}]}."
)
_STAFF_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_staff",
        "description": "Emit extracted staff objects aligned across Name(s) and Role(s).",
        "parameters": {
            "type": "object",
            "properties": {
                "people": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {"type": "string"},
                            "roles": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            },
            "required": ["people"]
        }
    }
}

def parse_staff_with_llm(block_text: str, model: str = "gpt-5") -> List[Dict[str, Any]]:
    if not isinstance(block_text, str) or not block_text.strip():
        return []
    resp = client.chat.completions.create(
        model=model,
        service_tier="flex",
        messages=[{"role": "system", "content": _SYSTEM_MSG},
                  {"role": "user", "content": block_text}],
        tools=[_STAFF_TOOL],
        temperature=0,
    )
    ch = resp.choices[0]
    if getattr(ch.message, "tool_calls", None):
        try:
            data = json.loads(ch.message.tool_calls[0].function.arguments)
            return data.get("people", []) or []
        except Exception:
            return []
    try:
        data = json.loads(ch.message.content or "{}")
        return data.get("people", []) or []
    except Exception:
        return []

# =========================
# 5) OFFLINE GENDER
# =========================

import gender_guesser.detector as gender
_detector = gender.Detector(case_sensitive=False)

def infer_gender_offline(name: Optional[str]) -> str:
    if not isinstance(name, str) or not name.strip():
        return "unknown"
    result = _detector.get_gender(name.strip().split()[0])
    mapping = {
        "male": "male",
        "mostly_male": "male",
        "female": "female",
        "mostly_female": "female",
        "andy": "unknown",
        "unknown": "unknown",
    }
    return mapping.get(result, "unknown")

# =========================
# 6) PIPELINE ENTRY POINT
# =========================

def get_staff_rows(
    input_csv_path="../data/final/enhanced_ref_data.csv",
    out_dir="../data/ics_staff_rows",
    base_url="https://results2021.ref.ac.uk/impact",
    model_staff="gpt-5",
    sleep_between_calls=0.03
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)

    df_ids = pd.read_csv(input_csv_path)
    ids = df_ids["REF impact case study identifier"].astype(str).tolist()

    all_texts = {}
    for ics in tqdm(ids, desc="Downloading & extracting PDFs"):
        target = f"{base_url}/{ics}/pdf"
        try:
            r = requests.get(target, timeout=60)
            r.raise_for_status()
            text = extract_text_safe(r.content)
            all_texts[ics] = text
        except Exception:
            all_texts[ics] = None

    df_texts = pd.DataFrame(list(all_texts.items()),
                            columns=["REF impact case study identifier", "Extracted Text"])
    df_texts.to_csv(os.path.join(out_dir, "ref_case_studies_text.csv"), index=False)

    # Compute staff blocks for all rows (do NOT drop rows with no block)
    df_staff_blocks = (
        df_texts.assign(staff_block=lambda d: d["Extracted Text"].apply(isolate_staff_names_block))
                .reset_index(drop=True)
    )
    df_staff_blocks.to_csv(os.path.join(out_dir, "staff_blocks.csv"), index=False)

    records: List[Dict[str, Any]] = []
    # Iterate over ALL rows, writing a row of NaNs if nothing is extracted
    for _, r in tqdm(df_staff_blocks.iterrows(), total=len(df_staff_blocks), desc="Extracting staff with LLM"):
        ics_id, block = r["REF impact case study identifier"], r["staff_block"]
        people: List[Dict[str, Any]] = []
        if isinstance(block, str) and block.strip():
            try:
                people = parse_staff_with_llm(block, model=model_staff)
            except Exception as e:
                people = [{"name": None, "roles": [], "error": str(e)}]

        # If nothing extracted, still write a placeholder row of NaNs to preserve row counts
        if not people:
            records.append({
                "REF impact case study identifier": ics_id,
                "name": pd.NA,
                "name_no_titles": pd.NA,
                "given_name": pd.NA,
                "role": pd.NA
            })
            time.sleep(sleep_between_calls)
            continue

        for person in people:
            raw_name = (person.get("name") or "").strip()
            name_norm = normalize_name(raw_name)
            name_no_titles = strip_titles(name_norm)
            given_name = extract_given_name(name_no_titles)
            roles = [x.strip() for x in (person.get("roles") or []) if x.strip()]
            records.append({
                "REF impact case study identifier": ics_id,
                "name": name_norm or pd.NA,
                "name_no_titles": name_no_titles or pd.NA,
                "given_name": given_name or pd.NA,
                "role": "; ".join(roles) if roles else pd.NA
            })
        time.sleep(sleep_between_calls)

    df_staff_rows = pd.DataFrame.from_records(records)
    df_staff_rows["offline_gender"] = df_staff_rows["given_name"].apply(infer_gender_offline)
    df_staff_rows.to_csv(os.path.join(out_dir, "ref_staff_rows.csv"), index=False)

    # Aggregate (preserve NaNs; do not fill with empty strings)
    df = df_staff_rows.copy()
    grouped = (
        df.groupby("REF impact case study identifier", dropna=False)
          .agg(
              names=("name", list),
              given_names=("given_name", list),
              roles=("role", list),
              genders=("offline_gender", list)
          )
    )
    counts = (
        df.groupby("REF impact case study identifier", dropna=False)["offline_gender"]
          .value_counts().unstack(fill_value=0)
          .rename(columns={
              "male": "number_male",
              "female": "number_female",
              "unknown": "number_unknown"
          })
    )
    # Ensure expected columns exist even if missing in the data
    for col in ("number_male", "number_female", "number_unknown"):
        if col not in counts.columns:
            counts[col] = 0

    ref_case_level = grouped.join(counts, how="left").fillna(0).reset_index()
    ref_case_level["number_people"] = (
        ref_case_level[["number_male", "number_female", "number_unknown"]].sum(axis=1).astype(int)
    )
    ref_case_level = ref_case_level[[
        "REF impact case study identifier",
        "names", "given_names", "roles", "genders",
        "number_people", "number_male", "number_female", "number_unknown"
    ]]
    ref_case_level.to_csv(os.path.join(out_dir, "ref_case_level.csv"), index=False)

    # Assert that output row counts align with the original number of rows
    n_original = len(df_ids)
    n_unique_staff_rows = df_staff_rows["REF impact case study identifier"].nunique()
    n_case_level = len(ref_case_level)
    assert n_unique_staff_rows == n_original, (
        f"Mismatch: unique staff rows ({n_unique_staff_rows}) != original rows ({n_original})"
    )
    assert n_case_level == n_original, (
        f"Mismatch: aggregated case rows ({n_case_level}) != original rows ({n_original})"
    )

    return df_staff_rows, ref_case_level

if __name__ == "__main__":
    rows, cases = get_staff_rows()
    print(rows.head())
    print(cases.head())
