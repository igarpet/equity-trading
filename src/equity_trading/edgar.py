"""SEC EDGAR fundamentals: bulk download and point-in-time parsing (Fair_Value branch, FV 01).

Sources (free, no key; SEC requires a descriptive User-Agent, set SEC_USER_AGENT in .env,
e.g. "Ivan Garcia Petrov ivan@example.com"; fair-access limit 10 requests/second):

- companyfacts.zip   every XBRL numeric fact of every filer, each with the filing that reported it
                     (accn, form, filed). Restated values appear again under the later filing, so
                     keeping every row and filtering on ``filed <= t`` gives a point-in-time panel.
- submissions.zip    per-CIK metadata (name, SIC, current tickers, former names) and the full
                     filing list with acceptance timestamps and primary document names.
- Financial Statement Data Sets (quarterly, 2009q1 ->): only ``sub.txt`` is kept. Its ``instance``
                     file name (e.g. ``aapl-20130928.xml``) carries the filer's ticker-like prefix,
                     which is how delisted CIKs get a historical symbol.

Layout under <EQUITY_TRADING_DIR>/EDGAR/:
  raw/companyfacts_<YYYYMMDD>.zip, raw/submissions_<YYYYMMDD>.zip, raw/company_tickers_exchange_<YYYYMMDD>.json
  raw/fsds/<YYYYqN>_sub.parquet
  facts/part-<NNNNN>.parquet         long facts table (see FACT_COLUMNS)
  companies.parquet, filings.parquet, fsds_sub.parquet, ticker_history.parquet

Nothing here drops data by date: EDGAR bulk files include 2026 filings. Studies cut at
``filed <= 2025-12-31`` (2026 sealed, repository rule).
"""
from __future__ import annotations

import io
import json
import os
import re
import time
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .data import vendor_dir

EDGAR_VERSION = "0.1.2"

COMPANYFACTS_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
SUBMISSIONS_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
FSDS_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip"

# Periodic reports kept in the facts table (amendments kept: they are new information on their own filed date).
FACT_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-KT/A", "10-QT", "10-QT/A",
                        "20-F", "20-F/A", "40-F", "40-F/A"})
FACT_TAXONOMIES = ("us-gaap", "dei", "ifrs-full")
FACT_COLUMNS = ["cik", "taxonomy", "tag", "unit", "start", "end", "val", "accn", "fy", "fp", "form",
                "filed", "frame"]

# Filing-list forms kept: periodic reports, events (8-K: M&A, guidance), deregistration / delisting,
# merger proxies and tender offers (deal prices for acquired firms).
FILING_FORMS = frozenset(FACT_FORMS | {"8-K", "8-K/A", "6-K", "15-12B", "15-12G", "15-15D", "15F-12B",
                                       "15F-12G", "15F-15D", "25", "25-NSE", "DEFM14A", "DEFM14C",
                                       "PREM14A", "SC TO-T", "SC 14D9", "S-4", "424B3"})
DELIST_FORMS = frozenset({"15-12B", "15-12G", "15-15D", "15F-12B", "15F-12G", "15F-15D", "25", "25-NSE"})

FSDS_SUB_COLUMNS = ["adsh", "cik", "name", "sic", "countryba", "countryinc", "afs", "fye", "form", "period",
                    "fy", "fp", "filed", "accepted", "prevrpt", "instance"]

# EDGAR Filer Manual naming: "<prefix>-<YYYYMMDD>.<ext>", prefix recommended to be the ticker.
_PREFIX_RE = re.compile(r"^([a-z]{1,5})-(\d{8})(?:[._]|$)")


# ----------------------------------------------------------------------------------------------- paths

def edgar_dir() -> Path:
    return vendor_dir("EDGAR")


def raw_dir() -> Path:
    d = edgar_dir() / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def latest_raw(stem: str, ext: str = "zip") -> Path:
    """Most recent dated snapshot raw/<stem>_<YYYYMMDD>.<ext>."""
    files = sorted(raw_dir().glob(f"{stem}_[0-9]*.{ext}"))
    if not files:
        raise FileNotFoundError(f"no {stem}_*.{ext} in {raw_dir()}: run tools/edgar/edgar_download.py first")
    return files[-1]


# ----------------------------------------------------------------------------------------------- http

def user_agent() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua or "@" not in ua:
        raise RuntimeError("SEC_USER_AGENT not set: add e.g. SEC_USER_AGENT=\"Your Name you@mail.com\" to .env "
                           "(SEC fair-access policy requires a name and contact e-mail).")
    return ua


def _session():
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent(), "Accept-Encoding": "gzip, deflate"})
    return s


def download_file(url: str, dest: Path, session=None, retries: int = 4, chunk: int = 1 << 20) -> Path:
    """Stream url to dest via dest.part (an interrupted transfer never leaves a truncated final file)."""
    s = session or _session()
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with s.get(url, stream=True, timeout=120) as r:
                if r.status_code == 404:
                    raise FileNotFoundError(url)
                r.raise_for_status()
                # Content-Length is the size on the wire: with gzip/deflate encoding requests decompresses
                # while streaming, so the byte count can only be checked for unencoded transfers.
                encoded = r.headers.get("Content-Encoding", "identity").lower() not in ("", "identity")
                total = 0 if encoded else int(r.headers.get("Content-Length", 0))
                done = 0
                with open(part, "wb") as f:
                    for b in r.iter_content(chunk):
                        f.write(b)
                        done += len(b)
                if total and done != total:
                    raise IOError(f"{url}: got {done} of {total} bytes")
            part.replace(dest)
            return dest
        except FileNotFoundError:
            raise
        except Exception as e:  # noqa: BLE001 — network errors: back off and retry
            if attempt == retries - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"  {type(e).__name__}: {e} — retry in {wait}s")
            time.sleep(wait)
    return dest


def download_bulk(force: bool = False, stamp: str | None = None) -> dict[str, Path]:
    """Download today's companyfacts.zip, submissions.zip and company_tickers_exchange.json (dated snapshots).

    Skips a file whose snapshot for ``stamp`` (default today) already exists, unless force.
    companyfacts.zip and submissions.zip are each ~1-2 GB.
    """
    stamp = stamp or date.today().strftime("%Y%m%d")
    s = _session()
    out = {}
    for stem, url, ext in [("company_tickers_exchange", TICKERS_URL, "json"),
                           ("submissions", SUBMISSIONS_URL, "zip"),
                           ("companyfacts", COMPANYFACTS_URL, "zip")]:
        dest = raw_dir() / f"{stem}_{stamp}.{ext}"
        if dest.exists() and not force:
            print(f"{dest.name}: exists, skipped")
        else:
            t0 = time.time()
            print(f"{dest.name}: downloading {url}")
            download_file(url, dest, s)
            print(f"  {dest.stat().st_size / 1e6:,.0f} MB in {time.time() - t0:,.0f}s")
        out[stem] = dest
    return out


def fsds_quarters(first: str = "2009q1", last: str | None = None) -> list[str]:
    """Quarter labels 2009q1 .. last (default: the last completed quarter)."""
    y0, q0 = int(first[:4]), int(first[-1])
    if last is None:
        t = date.today()
        q = (t.month - 1) // 3  # completed quarters this year
        y1, q1 = (t.year, q) if q else (t.year - 1, 4)
    else:
        y1, q1 = int(last[:4]), int(last[-1])
    out = []
    y, q = y0, q0
    while (y, q) <= (y1, q1):
        out.append(f"{y}q{q}")
        y, q = (y, q + 1) if q < 4 else (y + 1, 1)
    return out


def download_fsds(quarters: list[str] | None = None, keep_zip: bool = False) -> list[Path]:
    """Download Financial Statement Data Sets quarter zips and keep only sub.txt (as parquet).

    Resumable: a quarter whose raw/fsds/<q>_sub.parquet exists is skipped. SEC publishes a quarter's
    file a few weeks after it ends; a missing latest quarter is reported and skipped.
    """
    d = raw_dir() / "fsds"
    d.mkdir(exist_ok=True)
    s = _session()
    out = []
    for q in quarters or fsds_quarters():
        pq = d / f"{q}_sub.parquet"
        if pq.exists():
            out.append(pq)
            continue
        z = d / f"{q}.zip"
        try:
            if not z.exists():
                download_file(FSDS_URL.format(q=q), z, s)
        except FileNotFoundError:
            print(f"{q}: not published, skipped")
            continue
        with zipfile.ZipFile(z) as zf:
            with zf.open("sub.txt") as f:
                sub = read_fsds_sub(f)
        sub.to_parquet(pq, index=False)
        if not keep_zip:
            z.unlink()
        print(f"{q}: {len(sub):,} submissions")
        out.append(pq)
        time.sleep(0.2)
    return out


def read_fsds_sub(f) -> pd.DataFrame:
    """Parse an FSDS sub.txt (tab separated) into typed columns."""
    sub = pd.read_csv(f, sep="\t", dtype=str, usecols=lambda c: c in FSDS_SUB_COLUMNS,
                      encoding="utf-8", encoding_errors="replace", quoting=3, keep_default_na=False)
    sub["cik"] = sub["cik"].astype("int64")
    sub["filed"] = pd.to_datetime(sub["filed"], format="%Y%m%d", errors="coerce")
    sub["period"] = pd.to_datetime(sub["period"], format="%Y%m%d", errors="coerce")
    sub["accepted"] = pd.to_datetime(sub["accepted"], errors="coerce")
    sub["fy"] = _fiscal_year(sub["fy"])
    sub["prevrpt"] = sub["prevrpt"].eq("1")
    sub["accn"] = sub.pop("adsh")
    return sub


# ----------------------------------------------------------------------------------------------- companyfacts

def facts_rows(doc: dict, forms=FACT_FORMS, taxonomies=FACT_TAXONOMIES) -> dict[str, list]:
    """Flatten one companyfacts JSON document into column lists (FACT_COLUMNS)."""
    cols: dict[str, list] = {c: [] for c in FACT_COLUMNS}
    cik = int(doc.get("cik", 0))
    for tax, tags in (doc.get("facts") or {}).items():
        if tax not in taxonomies:
            continue
        for tag, body in tags.items():
            for unit, obs in (body.get("units") or {}).items():
                for o in obs:
                    form = o.get("form")
                    if form not in forms:
                        continue
                    cols["cik"].append(cik)
                    cols["taxonomy"].append(tax)
                    cols["tag"].append(tag)
                    cols["unit"].append(unit)
                    cols["start"].append(o.get("start"))
                    cols["end"].append(o.get("end"))
                    cols["val"].append(o.get("val"))
                    cols["accn"].append(o.get("accn"))
                    cols["fy"].append(o.get("fy"))
                    cols["fp"].append(o.get("fp"))
                    cols["form"].append(form)
                    cols["filed"].append(o.get("filed"))
                    cols["frame"].append(o.get("frame"))
    return cols


def _fiscal_year(x: pd.Series) -> pd.Series:
    """Filer-reported fiscal year as Int16; typos outside 1900-2100 (e.g. 20151) and non-integers -> NA."""
    v = pd.to_numeric(x, errors="coerce")
    ok = v.between(1900, 2100) & (v == np.floor(v))
    return v.where(ok).astype("Int16")


def _facts_frame(cols: dict[str, list]) -> pd.DataFrame:
    df = pd.DataFrame(cols, columns=FACT_COLUMNS)
    df["cik"] = df["cik"].astype("int64")
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d", errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce").astype("float64")
    df["fy"] = _fiscal_year(df["fy"])
    for c in ("taxonomy", "tag", "unit", "fp", "form"):
        df[c] = df[c].astype("category")
    df["accn"] = df["accn"].astype("string")
    df["frame"] = df["frame"].astype("string")
    return df


def parse_companyfacts(zip_path: Path | None = None, out_dir: Path | None = None, per_part: int = 500,
                       forms=FACT_FORMS, taxonomies=FACT_TAXONOMIES, overwrite: bool = False) -> pd.DataFrame:
    """companyfacts.zip -> facts/part-NNNNN.parquet, ``per_part`` companies per part, resumable.

    Members are processed in sorted name order, so part numbers are stable for a given snapshot.
    A manifest (facts/manifest.json) records the source snapshot; parsing a different snapshot into
    an existing folder requires overwrite=True. Returns a per-part summary.
    """
    zip_path = zip_path or latest_raw("companyfacts")
    out_dir = out_dir or edgar_dir() / "facts"
    out_dir.mkdir(parents=True, exist_ok=True)
    man = out_dir / "manifest.json"
    meta = {"source": zip_path.name, "version": EDGAR_VERSION, "per_part": per_part,
            "forms": sorted(forms), "taxonomies": list(taxonomies)}
    if man.exists():
        old = json.loads(man.read_text())
        if (old.get("source"), old.get("per_part")) != (meta["source"], per_part):
            if not overwrite:
                raise RuntimeError(f"{out_dir} holds {old.get('source')}; pass overwrite=True to reparse {zip_path.name}")
            for p in out_dir.glob("part-*.parquet"):
                p.unlink()
    man.write_text(json.dumps(meta, indent=1))
    summary = []
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".json"))
        for k in range(0, len(names), per_part):
            part = out_dir / f"part-{k // per_part:05d}.parquet"
            if part.exists():
                continue
            t0 = time.time()
            acc: dict[str, list] = {c: [] for c in FACT_COLUMNS}
            for n in names[k:k + per_part]:
                doc = json.loads(zf.read(n))
                rows = facts_rows(doc, forms, taxonomies)
                for c in FACT_COLUMNS:
                    acc[c].extend(rows[c])
            df = _facts_frame(acc)
            df.to_parquet(part, index=False)
            summary.append({"part": part.name, "companies": len(names[k:k + per_part]), "rows": len(df),
                            "seconds": round(time.time() - t0, 1)})
            print(f"{part.name}: {summary[-1]['companies']} companies, {len(df):,} facts, {summary[-1]['seconds']}s")
    return pd.DataFrame(summary)


def load_facts(tags: list[str] | None = None, ciks: list[int] | None = None, filed_max: str | None = "2025-12-31",
               facts_dir: Path | None = None) -> pd.DataFrame:
    """Read the facts table, optionally filtered by tag, CIK and filing date (default: 2026 sealed)."""
    import pyarrow.dataset as ds
    files = sorted(str(p) for p in Path(facts_dir or edgar_dir() / "facts").glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError("no facts parts: run tools/edgar/edgar_download.py --steps facts")
    d = ds.dataset(files, format="parquet")
    f = None
    if tags is not None:
        f = ds.field("tag").isin(tags)
    if ciks is not None:
        g = ds.field("cik").isin([int(c) for c in ciks])
        f = g if f is None else f & g
    if filed_max is not None:
        g = ds.field("filed") <= pd.Timestamp(filed_max)
        f = g if f is None else f & g
    return d.to_table(filter=f).to_pandas()


# ----------------------------------------------------------------------------------------------- submissions

_FILING_FIELDS = ["accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "form", "isXBRL",
                  "isInlineXBRL", "primaryDocument"]


def _filings_block(block: dict, cik: int, forms) -> pd.DataFrame:
    n = len(block.get("accessionNumber", []))
    if not n:
        return pd.DataFrame()
    df = pd.DataFrame({f: block.get(f, [None] * n) for f in _FILING_FIELDS})
    df.insert(0, "cik", cik)
    return df[df["form"].isin(forms)] if forms is not None else df


def parse_submissions(zip_path: Path | None = None, forms=FILING_FORMS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """submissions.zip -> (companies, filings). Reads both CIK##########.json and the
    CIK##########-submissions-NNN.json overflow files (older filings). Writes companies.parquet and
    filings.parquet to EDGAR/.
    """
    zip_path = zip_path or latest_raw("submissions")
    comp, fil = [], []
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".json"))
        for i, n in enumerate(names):
            doc = json.loads(zf.read(n))
            if "-submissions-" in n:
                cik = int(re.search(r"CIK(\d{10})", n).group(1))
                fil.append(_filings_block(doc, cik, forms))
                continue
            cik = int(doc.get("cik", 0))
            comp.append({
                "cik": cik, "name": doc.get("name"), "entity_type": doc.get("entityType"),
                "sic": doc.get("sic") or None, "sic_description": doc.get("sicDescription"),
                "category": doc.get("category"), "fiscal_year_end": doc.get("fiscalYearEnd"),
                "state_of_incorporation": doc.get("stateOfIncorporation"),
                "tickers": "|".join(t for t in doc.get("tickers") or [] if t),
                "exchanges": "|".join(e or "" for e in doc.get("exchanges") or []),
                "former_names": json.dumps(doc.get("formerNames") or []),
            })
            fil.append(_filings_block((doc.get("filings") or {}).get("recent") or {}, cik, forms))
            if (i + 1) % 100_000 == 0:
                print(f"  {i + 1:,} of {len(names):,} files")
    companies = pd.DataFrame(comp)
    filings = pd.concat([f for f in fil if len(f)], ignore_index=True)
    filings = filings.rename(columns={"accessionNumber": "accn", "filingDate": "filing_date",
                                      "reportDate": "report_date", "acceptanceDateTime": "accepted",
                                      "isXBRL": "is_xbrl", "isInlineXBRL": "is_inline_xbrl",
                                      "primaryDocument": "primary_document"})
    filings["filing_date"] = pd.to_datetime(filings["filing_date"], errors="coerce")
    filings["report_date"] = pd.to_datetime(filings["report_date"], errors="coerce")
    # acceptanceDateTime is ISO with a trailing Z but is US Eastern wall-clock time (EDGAR convention).
    filings["accepted"] = pd.to_datetime(filings["accepted"].astype(str).str.replace("Z", "", regex=False)
                                         .str.slice(0, 19), errors="coerce")
    for c in ("is_xbrl", "is_inline_xbrl"):
        filings[c] = filings[c].fillna(0).astype(bool)
    filings["form"] = filings["form"].astype("category")
    filings = filings.drop_duplicates("accn").reset_index(drop=True)
    companies.to_parquet(edgar_dir() / "companies.parquet", index=False)
    filings.to_parquet(edgar_dir() / "filings.parquet", index=False)
    return companies, filings


def combine_fsds() -> pd.DataFrame:
    """Concatenate raw/fsds/*_sub.parquet into EDGAR/fsds_sub.parquet (one row per accession)."""
    parts = sorted((raw_dir() / "fsds").glob("*_sub.parquet"))
    if not parts:
        raise FileNotFoundError("no FSDS quarters downloaded")
    sub = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    sub = sub.sort_values("filed").drop_duplicates("accn", keep="last").reset_index(drop=True)
    sub.to_parquet(edgar_dir() / "fsds_sub.parquet", index=False)
    return sub


# ----------------------------------------------------------------------------------------------- tickers

def file_prefix(name: str | None) -> str | None:
    """Ticker-like prefix of an XBRL instance / inline primary document name, or None.

    'aapl-20230930.htm' -> 'AAPL'; 'brkb-20231231_htm.xml' -> 'BRKB'; 'form10-k.htm' -> None.
    Heuristic: the Filer Manual recommends, not requires, the ticker as prefix.
    """
    if not name:
        return None
    m = _PREFIX_RE.match(str(name).strip().lower())
    if not m:
        return None
    d = m.group(2)
    if not ("1990" <= d[:4] <= "2099"):
        return None
    return m.group(1).upper()


def ticker_history(filings: pd.DataFrame | None = None, fsds_sub: pd.DataFrame | None = None,
                   companies: pd.DataFrame | None = None) -> pd.DataFrame:
    """Symbol candidates per CIK with the window in which each was used.

    Sources: ``fsds`` (instance names 2009+), ``primary`` (inline-XBRL primary documents, mostly 2019+),
    ``current`` (SEC company_tickers / submissions tickers; window = last periodic filing).
    Columns: cik, symbol, source, first_filed, last_filed, n_filings. Written to ticker_history.parquet.
    The prefix is not guaranteed to be the trading symbol; FV 01 notebook validates it against prices.
    """
    e = edgar_dir()
    filings = filings if filings is not None else pd.read_parquet(e / "filings.parquet")
    periodic = filings[filings["form"].isin(FACT_FORMS)]
    out = []
    if fsds_sub is None and (e / "fsds_sub.parquet").exists():
        fsds_sub = pd.read_parquet(e / "fsds_sub.parquet")
    if fsds_sub is not None and len(fsds_sub):
        s = fsds_sub.assign(symbol=fsds_sub["instance"].map(file_prefix)).dropna(subset=["symbol"])
        out.append(s[["cik", "symbol", "filed"]].assign(source="fsds"))
    p = periodic[periodic["is_inline_xbrl"]]
    p = p.assign(symbol=p["primary_document"].map(file_prefix)).dropna(subset=["symbol"])
    out.append(p[["cik", "symbol", "filing_date"]].rename(columns={"filing_date": "filed"}).assign(source="primary"))
    companies = companies if companies is not None else pd.read_parquet(e / "companies.parquet")
    last = periodic.groupby("cik")["filing_date"].agg(["min", "max"])
    cur = companies[companies["tickers"].fillna("") != ""][["cik", "tickers"]].copy()
    cur = cur.assign(symbol=cur["tickers"].str.split("|")).explode("symbol")
    cur = cur.join(last, on="cik").dropna(subset=["max"])
    out.append(pd.concat([cur[["cik", "symbol"]].assign(filed=cur["min"]),
                          cur[["cik", "symbol"]].assign(filed=cur["max"])]).assign(source="current"))
    h = pd.concat(out, ignore_index=True)
    h["symbol"] = h["symbol"].str.upper().str.strip()
    h = (h.groupby(["cik", "symbol", "source"])["filed"]
         .agg(first_filed="min", last_filed="max", n_filings="count").reset_index())
    h.to_parquet(e / "ticker_history.parquet", index=False)
    return h


def company_status(filings: pd.DataFrame | None = None, as_of: str = "2025-12-31") -> pd.DataFrame:
    """Per CIK: first/last periodic filing, deregistration/delisting form dates, and an 'active' flag
    (a periodic report filed within 15 months before as_of). Delisting dates are approximations:
    the last price date from the price data is the authoritative end of trading.
    """
    e = edgar_dir()
    filings = filings if filings is not None else pd.read_parquet(e / "filings.parquet")
    t = pd.Timestamp(as_of)
    f = filings[filings["filing_date"] <= t]
    per = f[f["form"].isin(FACT_FORMS)].groupby("cik")["filing_date"].agg(first_periodic="min", last_periodic="max",
                                                                          n_periodic="count")
    dl = f[f["form"].isin(DELIST_FORMS)].groupby("cik")["filing_date"].agg(first_delist_form="min")
    st = per.join(dl, how="left")
    st["active"] = st["last_periodic"] >= t - pd.DateOffset(months=15)
    return st.reset_index()


def yahoo_symbol(symbol: str) -> str:
    """EDGAR / exchange symbol -> Yahoo symbol (class shares use '-': BRK.B -> BRK-B)."""
    return symbol.upper().replace(".", "-").replace("/", "-")
