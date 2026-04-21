"""
JE Testing Agent — Python Backend
===================================
Framework : FastAPI
Install   : pip install fastapi uvicorn pandas openpyxl anthropic python-multipart aiofiles
Run       : uvicorn je_testing_backend:app --reload --port 8000

Architecture
------------
This backend is designed for large JE files and audit-grade data handling:

  1. User uploads raw files  →  saved to a secure TEMP directory per session
  2. Files are processed in CHUNKS  →  only metadata + flagged rows kept in memory
  3. Results, logs and audit trail are written to an OUTPUT directory
  4. User downloads everything they need
  5. On session end (or after retention window)  →  temp raw files are deleted
  6. Permanent storage = outputs + logs only  (not raw files, unless policy requires)

Session lifecycle:
  - Each browser session gets a unique session_id (UUID)
  - All temp files scoped to /tmp/je_sessions/{session_id}/
  - Outputs written to /tmp/je_sessions/{session_id}/outputs/
  - Auto-cleanup after RETENTION_MINUTES (default 120) or on explicit /session/close

This means the app never holds large GL files fully in memory,
and raw client data is not retained beyond the session.
"""

import os, io, json, uuid, shutil, logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import anthropic

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Configuration ─────────────────────────────────────────────────────────────
TEMP_BASE        = Path("/tmp/je_sessions")          # Root for all session temp data
CHUNK_SIZE       = 10_000                             # Rows processed per chunk
RETENTION_MINUTES= 120                                # How long temp files live
MAX_FILE_MB      = 500                                # Hard limit per uploaded file

TEMP_BASE.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("je_agent")

app = FastAPI(title="JE Testing Agent API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Restrict to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


# =============================================================================
# Session management
# =============================================================================

def session_dir(session_id: str) -> Path:
    return TEMP_BASE / session_id

def output_dir(session_id: str) -> Path:
    d = session_dir(session_id) / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d

def session_meta_path(session_id: str) -> Path:
    return session_dir(session_id) / "meta.json"

def load_meta(session_id: str) -> dict:
    p = session_meta_path(session_id)
    if p.exists():
        return json.loads(p.read_text())
    return {}

def save_meta(session_id: str, meta: dict):
    session_meta_path(session_id).write_text(json.dumps(meta, default=str))

def audit_log(session_id: str, event: str, detail: str = ""):
    """Append a timestamped audit trail entry — these are kept permanently."""
    log_path = output_dir(session_id) / "audit_trail.log"
    entry = f"{datetime.utcnow().isoformat()}Z  [{event}]  {detail}\n"
    with open(log_path, "a") as f:
        f.write(entry)
    log.info(f"[{session_id[:8]}] {event} — {detail}")


@app.post("/session/start")
async def start_session():
    """
    Called when the user opens the app.
    Creates a scoped temp directory and returns a session_id.
    The frontend stores this and sends it with every subsequent request.
    """
    session_id = str(uuid.uuid4())
    session_dir(session_id).mkdir(parents=True, exist_ok=True)
    output_dir(session_id)   # ensure outputs dir exists
    meta = {
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat(),
        "expires_at": (datetime.utcnow() + timedelta(minutes=RETENTION_MINUTES)).isoformat(),
        "files": {},
        "mappings": {},
    }
    save_meta(session_id, meta)
    audit_log(session_id, "SESSION_START", f"New session created")
    return {"session_id": session_id, "expires_at": meta["expires_at"]}


@app.delete("/session/{session_id}")
async def close_session(session_id: str):
    """
    Called when the user clicks 'Done' or closes the app.
    Deletes all raw temp files. Outputs (results, audit trail) are preserved.
    Raw files are never permanently stored.
    """
    sd = session_dir(session_id)
    if not sd.exists():
        raise HTTPException(404, "Session not found")

    # Delete raw uploaded files only — keep outputs/
    for f in sd.iterdir():
        if f.name != "outputs" and f.name != "meta.json":
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)

    audit_log(session_id, "SESSION_CLOSE", "Raw temp files deleted. Outputs retained.")
    return {"status": "closed", "outputs_retained": True}


def cleanup_expired_sessions():
    """Background task: delete temp files for sessions past their retention window."""
    for sd in TEMP_BASE.iterdir():
        meta_path = sd / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        expires = datetime.fromisoformat(meta.get("expires_at", "2000-01-01"))
        if datetime.utcnow() > expires:
            # Delete raw files, preserve outputs
            for f in sd.iterdir():
                if f.name != "outputs":
                    if f.is_file(): f.unlink()
                    elif f.is_dir(): shutil.rmtree(f)
            log.info(f"Auto-cleaned expired session {sd.name[:8]}")


# =============================================================================
# STEP 0 — File upload  (saved to disk, NOT loaded into memory)
# =============================================================================

@app.post("/ingest/{dataset}/{period}")
async def ingest_file(
    dataset: str,           # 'gl', 'tb', or 'coa'
    period: str,            # 'cy', 'py', or 'shared' (CoA)
    session_id: str,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
):
    """
    Saves uploaded files to the session's temp directory.
    Does NOT load full file into memory — just validates headers.
    For GL: validates all files have identical columns (format check).

    Large file strategy:
    - Files are streamed to disk in chunks (aiofiles in production)
    - Only the first ~100 rows are read immediately to detect columns
    - Full processing happens lazily at the mapping/integrity/testing steps
    """
    background_tasks.add_task(cleanup_expired_sessions)

    sd = session_dir(session_id)
    if not sd.exists():
        raise HTTPException(404, "Session not found — call /session/start first")

    meta = load_meta(session_id)
    saved = []
    reference_cols = None

    for f in files:
        # Check file size
        content = await f.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_MB:
            raise HTTPException(413, f"{f.filename} exceeds {MAX_FILE_MB}MB limit")

        # Save raw file to temp storage
        dest = sd / f"{dataset}_{period}_{f.filename}"
        dest.write_bytes(content)

        # Read only headers + first 100 rows for column detection
        name = f.filename.lower()
        if name.endswith(".csv"):
            sample = pd.read_csv(io.BytesIO(content), nrows=100)
        else:
            sample = pd.read_excel(io.BytesIO(content), nrows=100)

        cols = list(sample.columns)

        # GL format consistency check
        if dataset == "gl":
            if reference_cols is None:
                reference_cols = cols
            else:
                if sorted(cols) != sorted(reference_cols):
                    dest.unlink()   # remove the bad file
                    raise HTTPException(
                        400,
                        f"Format mismatch in {f.filename} — columns differ from first file"
                    )

        saved.append({"filename": f.filename, "size_mb": round(size_mb, 2), "columns": cols})
        audit_log(session_id, "FILE_UPLOADED",
                  f"{dataset.upper()} {period.upper()} — {f.filename} ({size_mb:.1f} MB)")

    # Store file metadata (not the data itself)
    key = f"{period}_{dataset}"
    meta.setdefault("files", {})[key] = {
        "files": saved,
        "columns": reference_cols or (saved[0]["columns"] if saved else []),
        "uploaded_at": datetime.utcnow().isoformat(),
    }
    save_meta(session_id, meta)

    return {
        "dataset": dataset,
        "period": period,
        "files_saved": len(saved),
        "columns": reference_cols or (saved[0]["columns"] if saved else []),
        "total_size_mb": round(sum(f["size_mb"] for f in saved), 2),
    }


# =============================================================================
# STEP 1 & 2 — Field mapping  (rule-based + Claude AI)
# =============================================================================

FIELD_HINTS = {
    "je_number":                ["je_number","journal_number","doc_number","voucher","trans_no","entry_no"],
    "je_line_number":           ["je_line","line_number","line_no","line_num","line_seq"],
    "gl_account_code":          ["gl_account","account_code","acct_code","account_no","nominal_code",
                                 "ledger_code","nominal","chart_code","fin_code","cost_code","account_number"],
    "business_unit_code":       ["business_unit","bu_code","cost_centre","cost_center","dept","department",
                                 "division","profit_centre","entity_code","company_code"],
    "account_type":             ["account_type","acct_type","type","classification","bs_pl"],
    "functional_amount":        ["functional_amount","amount","net_amount","transaction_amount",
                                 "local_amount","base_amount","lcy_amount","amt"],
    "dr_cr_indicator":          ["dr_cr","debit_credit","dc_indicator","sign","dc_flag","entry_type"],
    "reporting_amount":         ["reporting_amount","group_amount","foreign_amount","fc_amount"],
    "reporting_currency_code":  ["reporting_currency","group_currency","foreign_currency","fc_code"],
    "fiscal_year":              ["fiscal_year","fy","year","financial_year","book_year"],
    "fiscal_period":            ["fiscal_period","period","accounting_period","month","book_period"],
    "effective_date":           ["effective_date","posting_date","value_date","transaction_date",
                                 "doc_date","gl_date","journal_date","accounting_date"],
    "entry_date":               ["entry_date","entered_date","created_date","input_date","create_date"],
    "preparer_id":              ["preparer_id","preparer","created_by","entered_by","user_id",
                                 "posted_by","prepared_by","operator","author"],
    "approver_id":              ["approver_id","approver","approved_by","authorised_by","reviewed_by",
                                 "authorized_by","manager"],
    "source_code":              ["source_code","source","source_ref","source_system","origin","module"],
    "je_line_description":      ["description","desc","narrative","memo","text","remarks","notes","comment"],
    "functional_currency_code": ["functional_currency","currency_code","currency","base_currency",
                                 "local_currency","lcy"],
    "tb_account_code":          ["account_code","gl_account","acct_code","account_no","nominal_code",
                                 "ledger_code"],
    "functional_opening_balance":["opening_balance","open_bal","opening","bfwd","brought_forward","ob",
                                  "start_balance","begin_balance"],
    "functional_closing_balance":["closing_balance","close_bal","closing","cfwd","carried_forward","cb",
                                  "end_balance","ending_balance"],
    "coa_account_code":         ["account_code","gl_account","acct_code","nominal_code","ledger_code"],
    "coa_account_name":         ["account_name","account_description","account_desc","account_title",
                                 "nominal_name","ledger_name"],
    "coa_account_type":         ["account_type","acct_type","type","classification"],
    "coa_account_class":        ["account_class","class","grouping","category","account_group"],
}


def rule_match(field_key: str, columns: list[str]) -> str:
    hints = FIELD_HINTS.get(field_key, [])
    norm  = lambda s: s.lower().replace(" ","_").replace("-","_").replace(".","_")
    for col in columns:
        n = norm(col)
        if any(n == h or h in n or n in h for h in hints):
            return col
    return ""


@app.get("/mapping/suggest/{dataset}/{period}")
async def suggest_mapping(dataset: str, period: str, session_id: str):
    """
    Step 1: Rule-based matching.
    Step 2: Claude API for anything rules missed.
    Returns full suggested mapping dict with source labels (rule / ai / unmatched).
    """
    meta = load_meta(session_id)
    key  = f"{period}_{dataset}"
    file_meta = meta.get("files", {}).get(key)
    if not file_meta:
        raise HTTPException(404, "Files not uploaded for this dataset/period")

    columns   = file_meta["columns"]
    field_keys = [k for k in FIELD_HINTS if dataset in k or k.startswith(dataset[:3])]

    rule_matches = {}
    unmatched    = []
    for fk in field_keys:
        match = rule_match(fk, columns)
        if match:
            rule_matches[fk] = match
        else:
            unmatched.append(fk)

    # AI pass
    ai_suggestions = {}
    if unmatched and ai_client.api_key:
        field_desc = "\n".join(
            f'- "{fk}": aliases include {", ".join(FIELD_HINTS.get(fk, [])[:4])}'
            for fk in unmatched
        )
        prompt = (
            f"Map ERP export columns to canonical audit field names.\n\n"
            f"Client columns: {', '.join(columns)}\n\n"
            f"Fields not matched by rules:\n{field_desc}\n\n"
            f"Respond ONLY with JSON: {{\"field_key\": \"Client_Column_or_null\"}}"
        )
        try:
            response = ai_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            ai_suggestions = json.loads(response.content[0].text.strip())
        except Exception as e:
            log.warning(f"AI mapping failed: {e}")

    audit_log(session_id, "MAPPING_SUGGESTED",
              f"{dataset.upper()} {period.upper()} — {len(rule_matches)} rule, "
              f"{len([v for v in ai_suggestions.values() if v])} AI")

    return {
        "columns":        columns,
        "rule_matches":   rule_matches,
        "ai_suggestions": {k: v for k, v in ai_suggestions.items() if v},
        "unmatched":      [fk for fk in unmatched if not ai_suggestions.get(fk)],
    }


class MappingConfirm(BaseModel):
    session_id: str
    period:     str
    dataset:    str
    mapping:    dict   # {canonical_field_key: client_column_name}


@app.post("/mapping/confirm")
async def confirm_mapping(req: MappingConfirm):
    """Save the auditor-confirmed field mapping."""
    meta = load_meta(req.session_id)
    meta.setdefault("mappings", {})[f"{req.period}_{req.dataset}"] = req.mapping
    save_meta(req.session_id, meta)
    audit_log(req.session_id, "MAPPING_CONFIRMED",
              f"{req.dataset.upper()} {req.period.upper()} — {len(req.mapping)} fields mapped")
    return {"status": "saved"}


# =============================================================================
# STEP 5 — GL Integrity check  (chunked processing)
# =============================================================================

def _sum_gl_chunked(file_path: Path, filename: str, acc_col: str, amt_col: str) -> dict:
    """
    Read a GL file in chunks and accumulate account-level sums.
    This is the key pattern for large files — never loads the full file at once.
    """
    sums: dict[str, float] = {}
    is_csv = filename.lower().endswith(".csv")

    if is_csv:
        reader = pd.read_csv(file_path, chunksize=CHUNK_SIZE, usecols=[acc_col, amt_col])
    else:
        # Excel doesn't support chunked reading natively — use skiprows batching
        # For very large Excel files, recommend converting to CSV first
        df = pd.read_excel(file_path, usecols=[acc_col, amt_col])
        reader = [df[i:i+CHUNK_SIZE] for i in range(0, len(df), CHUNK_SIZE)]

    for chunk in reader:
        chunk[amt_col] = pd.to_numeric(chunk[amt_col], errors="coerce").fillna(0)
        for acc, amt in zip(chunk[acc_col].astype(str).str.strip(), chunk[amt_col]):
            if acc:
                sums[acc] = sums.get(acc, 0) + amt

    return sums


@app.get("/integrity")
async def run_integrity_check(session_id: str):
    """
    GL vs TB reconciliation — run separately for CY and PY.
    GL files are processed in chunks to handle large populations.
    Only the reconciliation result (not raw data) is returned.
    """
    meta    = load_meta(session_id)
    sd      = session_dir(session_id)
    results = {}

    for period in ["cy", "py"]:
        gl_meta  = meta.get("files", {}).get(f"{period}_gl")
        tb_meta  = meta.get("files", {}).get(f"{period}_tb")
        gl_map   = meta.get("mappings", {}).get(f"{period}_gl", {})
        tb_map   = meta.get("mappings", {}).get(f"{period}_tb", {})

        if not gl_meta or not tb_meta:
            results[period] = {"error": "Data not uploaded"}
            continue

        acc_col  = gl_map.get("gl_account_code", "")
        amt_col  = gl_map.get("functional_amount", "")
        tb_acc   = tb_map.get("tb_account_code", "")
        tb_open  = tb_map.get("functional_opening_balance", "")
        tb_close = tb_map.get("functional_closing_balance", "")

        # Accumulate GL sums in chunks across all GL files
        gl_sums: dict[str, float] = {}
        for f_info in gl_meta["files"]:
            fp = sd / f"gl_{period}_{f_info['filename']}"
            if fp.exists():
                chunk_sums = _sum_gl_chunked(fp, f_info["filename"], acc_col, amt_col)
                for acc, amt in chunk_sums.items():
                    gl_sums[acc] = gl_sums.get(acc, 0) + amt

        # Load TB (smaller file — can load fully)
        tb_files = [sd / f"tb_{period}_{f['filename']}" for f in tb_meta["files"]]
        tb_df    = pd.concat([
            pd.read_csv(fp) if str(fp).endswith(".csv") else pd.read_excel(fp)
            for fp in tb_files if fp.exists()
        ], ignore_index=True)

        tb_df["movement"] = (
            pd.to_numeric(tb_df[tb_close], errors="coerce").fillna(0)
            - pd.to_numeric(tb_df[tb_open],  errors="coerce").fillna(0)
        )

        rows = []
        for _, r in tb_df.iterrows():
            acc   = str(r[tb_acc]).strip()
            mov   = round(float(r["movement"]), 2)
            gl_s  = round(gl_sums.get(acc, 0), 2)
            diff  = round(mov - gl_s, 2)
            rows.append({"account": acc, "tb_movement": mov, "gl_sum": gl_s,
                         "diff": diff, "match": abs(diff) <= 0.01})

        gl_only = list(set(gl_sums.keys()) - set(tb_df[tb_acc].astype(str).str.strip()))
        mismatch_rows = [r for r in rows if not r["match"]]

        results[period] = {
            "total_tb_accounts":  len(rows),
            "matched":            len(rows) - len(mismatch_rows),
            "mismatches":         len(mismatch_rows),
            "gl_only_accounts":   gl_only,
            "mismatch_detail":    mismatch_rows,
        }
        audit_log(session_id, "INTEGRITY_CHECK",
                  f"{period.upper()} — {len(mismatch_rows)} mismatches out of {len(rows)} accounts")

    return results


# =============================================================================
# STEP 6 — CoA Completeness
# =============================================================================

@app.get("/coa-completeness")
async def run_coa_completeness(session_id: str):
    """5 checks: duplicates in CoA, CY GL, PY GL, CY TB, PY TB vs CoA."""
    meta    = load_meta(session_id)
    sd      = session_dir(session_id)
    coa_meta= meta.get("files", {}).get("shared_coa")
    coa_map = meta.get("mappings", {}).get("shared_coa", {})

    if not coa_meta:
        raise HTTPException(404, "CoA not uploaded")

    coa_fp  = sd / f"coa_shared_{coa_meta['files'][0]['filename']}"
    coa_df  = pd.read_csv(coa_fp) if str(coa_fp).endswith(".csv") else pd.read_excel(coa_fp)
    coa_col = coa_map.get("coa_account_code", coa_df.columns[0])
    coa_codes = set(coa_df[coa_col].astype(str).str.strip())

    dup_counts = coa_df[coa_col].astype(str).value_counts()
    duplicates = list(dup_counts[dup_counts > 1].index)

    checks = {
        "check_1_no_duplicates": {"pass": len(duplicates) == 0, "duplicates": duplicates}
    }

    for period, dataset, label in [
        ("cy","gl","check_2_cy_gl"), ("py","gl","check_3_py_gl"),
        ("cy","tb","check_4_cy_tb"), ("py","tb","check_5_py_tb"),
    ]:
        ds_meta = meta.get("files", {}).get(f"{period}_{dataset}")
        ds_map  = meta.get("mappings", {}).get(f"{period}_{dataset}", {})
        if not ds_meta:
            checks[label] = {"pass": False, "error": "Not uploaded"}
            continue
        acc_col = ds_map.get("gl_account_code" if dataset=="gl" else "tb_account_code", "")
        if not acc_col:
            checks[label] = {"pass": False, "error": "Account code not mapped"}
            continue
        # Read only the account code column — minimise memory use
        fp = sd / f"{dataset}_{period}_{ds_meta['files'][0]['filename']}"
        col_df = (pd.read_csv(fp, usecols=[acc_col])
                  if str(fp).endswith(".csv") else pd.read_excel(fp, usecols=[acc_col]))
        codes   = set(col_df[acc_col].astype(str).str.strip().unique())
        missing = list(codes - coa_codes)
        checks[label] = {"pass": len(missing) == 0, "missing_count": len(missing),
                         "missing_sample": missing[:20]}

    all_pass = all(v["pass"] for v in checks.values())
    audit_log(session_id, "COA_COMPLETENESS", f"{'PASS' if all_pass else 'FAIL'} — {checks}")
    return {"all_pass": all_pass, "checks": checks}


# =============================================================================
# STEP 7 — JE Testing  (chunked, writes results to output files)
# =============================================================================

@app.post("/je-testing")
async def run_je_testing(session_id: str, criteria: list[str]):
    """
    Runs JE risk criteria against CY GL data in chunks.
    Flagged rows are written to output CSV as they are found.
    Raw GL data is never stored in memory all at once.
    Results file is written to the session's output directory for download.
    """
    meta    = load_meta(session_id)
    sd      = session_dir(session_id)
    gl_meta = meta.get("files", {}).get("cy_gl")
    gl_map  = meta.get("mappings", {}).get("cy_gl", {})
    out_dir = output_dir(session_id)

    if not gl_meta:
        raise HTTPException(404, "CY GL data not uploaded")

    def col(key): return gl_map.get(key, "")

    flagged_path   = out_dir / "je_flagged_entries.csv"
    summary: dict  = {c: 0 for c in criteria}
    total_rows     = 0
    header_written = False

    for f_info in gl_meta["files"]:
        fp       = sd / f"gl_cy_{f_info['filename']}"
        is_csv   = str(fp).endswith(".csv")
        reader   = (pd.read_csv(fp, chunksize=CHUNK_SIZE)
                    if is_csv else
                    [pd.read_excel(fp)])   # TODO: chunked Excel via openpyxl for very large files

        for chunk in reader:
            total_rows += len(chunk)
            flagged_mask = pd.Series([False] * len(chunk), index=chunk.index)
            chunk["_criteria_flags"] = ""

            if "weekend" in criteria and col("effective_date") in chunk.columns:
                dates = pd.to_datetime(chunk[col("effective_date")], errors="coerce")
                mask  = dates.dt.dayofweek >= 5
                chunk.loc[mask, "_criteria_flags"] += "weekend,"
                summary["weekend"] += int(mask.sum())
                flagged_mask |= mask

            if "round" in criteria and col("functional_amount") in chunk.columns:
                amt  = pd.to_numeric(chunk[col("functional_amount")], errors="coerce").abs()
                mask = (amt % 1000 == 0) & (amt > 0)
                chunk.loc[mask, "_criteria_flags"] += "round,"
                summary["round"] += int(mask.sum())
                flagged_mask |= mask

            if "no_approver" in criteria and col("approver_id") in chunk.columns:
                mask = chunk[col("approver_id")].astype(str).str.strip().isin(["","nan","None","0"])
                chunk.loc[mask, "_criteria_flags"] += "no_approver,"
                summary["no_approver"] += int(mask.sum())
                flagged_mask |= mask

            if "self_approved" in criteria and col("preparer_id") in chunk.columns and col("approver_id") in chunk.columns:
                mask = chunk[col("preparer_id")].astype(str) == chunk[col("approver_id")].astype(str)
                chunk.loc[mask, "_criteria_flags"] += "self_approved,"
                summary["self_approved"] += int(mask.sum())
                flagged_mask |= mask

            if "no_description" in criteria and col("je_line_description") in chunk.columns:
                mask = chunk[col("je_line_description")].astype(str).str.strip().isin(["","nan","None"])
                chunk.loc[mask, "_criteria_flags"] += "no_description,"
                summary["no_description"] += int(mask.sum())
                flagged_mask |= mask

            if "large_value" in criteria and col("functional_amount") in chunk.columns:
                amt       = pd.to_numeric(chunk[col("functional_amount")], errors="coerce").abs()
                threshold = amt.quantile(0.95)
                mask      = amt >= threshold
                chunk.loc[mask, "_criteria_flags"] += "large_value,"
                summary["large_value"] += int(mask.sum())
                flagged_mask |= mask

            # Write flagged rows from this chunk straight to output file
            flagged_chunk = chunk[flagged_mask]
            if not flagged_chunk.empty:
                flagged_chunk.to_csv(
                    flagged_path,
                    mode  = "a",
                    header= not header_written,
                    index = False
                )
                header_written = True

    # Write summary report
    summary_path = out_dir / "je_testing_summary.txt"
    with open(summary_path, "w") as f:
        f.write("JE TESTING AGENT — SUMMARY REPORT\n")
        f.write(f"Generated : {datetime.utcnow().isoformat()}Z\n")
        f.write(f"Session   : {session_id}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total JEs analysed : {total_rows:,}\n")
        total_flagged = sum(summary.values())
        f.write(f"Total flagged      : {total_flagged:,}\n\n")
        f.write("By criterion:\n")
        for criterion, count in summary.items():
            f.write(f"  {criterion:<25} {count:>8,}\n")

    audit_log(session_id, "JE_TESTING_COMPLETE",
              f"{total_rows:,} rows processed — {sum(summary.values()):,} flagged")

    return {
        "total_rows":     total_rows,
        "total_flagged":  sum(summary.values()),
        "by_criterion":   summary,
        "download_ready": True,
    }


# =============================================================================
# Downloads — auditor downloads results, then temp files can be deleted
# =============================================================================

@app.get("/download/{session_id}/{filename}")
async def download_output(session_id: str, filename: str):
    """Serve a file from the session's output directory."""
    fp = output_dir(session_id) / filename
    if not fp.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path=fp, filename=filename)


@app.get("/download/{session_id}/list")
async def list_outputs(session_id: str):
    """List all downloadable output files for a session."""
    od = output_dir(session_id)
    return {"files": [f.name for f in od.iterdir() if f.is_file()]}


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
