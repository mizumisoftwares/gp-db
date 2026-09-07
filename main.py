"""
FastAPI backend over DuckDB - Multi-repo distributed search API.

Features:
  - Streams Parquet files directly via HTTP Range Requests without downloading to disk.
  - Automatic index-aware routing for exact match acceleration.
  - Thread-confined DuckDB connection pool for high concurrency.
  - Custom deduplication capping (max 2 entries per person identity).
"""

import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Hugging Face Repositories Configuration ---
DATA_REPO_URL = "https://huggingface.co/datasets/Kzr0xx/Icmr-and-hitek/resolve/main"
INDEX_REPO_URL = "https://huggingface.co/datasets/HiTeckGroup/HiTeckgroop/resolve/main"

# Remote Parquet URLs
PARQUET_FILES = [
    f"{DATA_REPO_URL}/part1.parquet",
    f"{DATA_REPO_URL}/part2a.parquet",
    f"{DATA_REPO_URL}/part2b_new.parquet",
]

IDX_PHONE = f"{INDEX_REPO_URL}/idx_phone.parquet"
IDX_AADHAR = f"{INDEX_REPO_URL}/idx_aadhar.parquet"
IDX_NAME = f"{INDEX_REPO_URL}/idx_name.parquet"

# Performance Tuning (Tuned for memory-constrained environments like Render Free Tier)
PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "4"))
THREADS_PER_CONN = int(os.environ.get("ICMR_THREADS_PER_CONN", "2"))
DUPLICATE_CAP = 2

SEARCH_FIELDS = [
    "name",
    "fathersName",
    "phoneNumber",
    "aadharNumber",
    "otherNumber",
    "address",
    "district",
    "pincode",
    "state",
    "town",
    "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

# --- Global Thread Pool & Connection State ---
_conns: List[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")

# Track index availability state globally
_INDEX_STATUS = {"phone": False, "aadhar": False, "name": False}


def _new_conn() -> duckdb.DuckDBPyConnection:
    """Initializes a new thread-confined DuckDB connection with HTTP support."""
    con = duckdb.connect()  # Pure in-memory
    con.execute("INSTALL parquet; LOAD parquet; INSTALL httpfs; LOAD httpfs;")

    # Memory cap safeguard for cloud deployment
    con.execute("SET memory_limit = '512MB';")

    # Authenticate via HF token if repos are private/gated
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        con.execute(f"CREATE SECRET hf_sec (TYPE HUGGINGFACE, TOKEN '{hf_token}');")

    # 1. Register base view from raw parquet data files
    raw_files_str = ", ".join(f"'{f}'" for f in PARQUET_FILES)
    con.execute(f"CREATE VIEW people AS SELECT * FROM read_parquet([{raw_files_str}])")

    # 2. Attach pre-sorted index views if remote index files are available
    for idx_url, view_name, key in [
        (IDX_PHONE, "people_phone", "phone"),
        (IDX_AADHAR, "people_aadhar", "aadhar"),
        (IDX_NAME, "people_name", "name"),
    ]:
        try:
            con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{idx_url}')")
            _INDEX_STATUS[key] = True
        except Exception:
            _INDEX_STATUS[key] = False

    con.execute(f"SET threads = {THREADS_PER_CONN}")
    return con


def _thread_id() -> int:
    tid = getattr(_thread_local, "id", None)
    if tid is None:
        with _conns_lock:
            tid = len(_conns)
            _thread_local.id = tid
    return tid


def _get_conn() -> duckdb.DuckDBPyConnection:
    ident = _thread_id()
    with _conns_lock:
        while len(_conns) <= ident:
            _conns.append(_new_conn())
    return _conns[ident]


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    pool.shutdown(wait=True)
    with _conns_lock:
        for con in _conns:
            try:
                con.close()
            except Exception:
                pass


app = FastAPI(
    title="GP Search Engine",
    description="Data is fully trained on gram panchayat database of public datasets.",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Identity & Deduplication Helpers ---
def _person_key(row: Dict[str, Any]) -> Tuple[str, ...]:
    """Generates unique key per identity for deduplication."""
    ph = (row.get("phoneNumber") or "").strip()
    ad = (row.get("aadharNumber") or "").strip()
    if ph or ad:
        return (ph, ad)
    return (row.get("name") or "").strip(), (row.get("fathersName") or "").strip()


def _cap_duplicates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Limits duplicate records per identity based on DUPLICATE_CAP."""
    seen: Dict[Tuple[str, ...], int] = {}
    out: List[Dict[str, Any]] = []
    for r in rows:
        k = _person_key(r)
        n = seen.get(k, 0)
        if n < DUPLICATE_CAP:
            seen[k] = n + 1
            out.append(r)
    return out


def _source_clause(source: Optional[str], params: List[Any]) -> str:
    if source and source in ("icmr", "hitek", "inddata"):
        src = "hitek" if source == "hitek" else source
        params.append(src)
        return " AND source = ?"
    return ""


def _run_field_search(
    field: str,
    value: str,
    mode: str,
    limit: int,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")

    view = "people"
    params: List[Any] = []

    # Index Routing for performance optimization
    if mode == "exact":
        if field == "phoneNumber" and _INDEX_STATUS["phone"]:
            view = "people_phone"
        elif field == "aadharNumber" and _INDEX_STATUS["aadhar"]:
            view = "people_aadhar"
        elif field == "name" and _INDEX_STATUS["name"]:
            view = "people_name"

        params.append(value)
        sql = (
            f"SELECT * FROM {view} WHERE {field} = ?"
            f"{_source_clause(source, params)} LIMIT {limit * DUPLICATE_CAP + 20}"
        )
    elif mode == "contains":
        if field == "name" and _INDEX_STATUS["name"]:
            view = "people_name"

        v_escaped = value.replace("%", r"\%").replace("_", r"\_")
        params.append(f"%{v_escaped}%")
        sql = (
            f"SELECT * FROM {view} WHERE {field} ILIKE ? ESCAPE '\\'"
            f"{_source_clause(source, params)} LIMIT {limit * DUPLICATE_CAP + 20}"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    con = _get_conn()
    rows = con.execute(sql, params).fetchall()
    cols = [d[0] for d in con.description]
    results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]

    return {
        "field": field,
        "value": value,
        "mode": mode,
        "count": len(results),
        "results": results,
    }


async def _parallel(queries: List[Tuple[str, str, str, int]]) -> List[Dict[str, Any]]:
    loop = asyncio.get_running_loop()

    async def one(t: Tuple[str, str, str, int]) -> Dict[str, Any]:
        field, value, mode, limit = t
        return await loop.run_in_executor(
            pool, _run_field_search, field, value, mode, limit
        )

    return await asyncio.gather(*[one(t) for t in queries])


async def _unified_search(q: str, limit: int, source: Optional[str] = None) -> Dict[str, Any]:
    q = q.strip()
    is_num = q.isdigit() and len(q) >= 8
    loop = asyncio.get_running_loop()

    if is_num:
        all_rows: List[Dict[str, Any]] = []
        searched: List[str] = []

        if _INDEX_STATUS["phone"]:
            r = await loop.run_in_executor(
                pool, _run_field_search, "phoneNumber", q, "exact", limit, source
            )
            all_rows.extend(r["results"])
            searched.append("phoneNumber")

        if not all_rows and _INDEX_STATUS["aadhar"]:
            r = await loop.run_in_executor(
                pool, _run_field_search, "aadharNumber", q, "exact", limit, source
            )
            all_rows.extend(r["results"])
            searched.append("aadharNumber")

        if not all_rows and _INDEX_STATUS["name"]:
            r = await loop.run_in_executor(
                pool, _run_field_search, "otherNumber", q, "exact", limit, source
            )
            all_rows.extend(r["results"])
            searched.append("otherNumber")

        all_rows = _cap_duplicates(all_rows)[:limit]
        return {
            "query": q,
            "searched_fields": searched or list(NUMBER_FIELDS),
            "dedup": DUPLICATE_CAP,
            "count": len(all_rows),
            "results": all_rows,
        }
    else:
        jobs = [
            loop.run_in_executor(
                pool, _run_field_search, "name", q, "contains", limit, source
            )
        ]
        if _INDEX_STATUS["name"]:
            jobs.append(
                loop.run_in_executor(
                    pool, _run_field_search, "fathersName", q, "contains", limit, source
                )
            )

        results = await asyncio.gather(*jobs)
        all_rows = []
        for r in results:
            all_rows.extend(r["results"])

        all_rows = _cap_duplicates(all_rows)[:limit]
        fields = ["name", "fathersName"] + NUMBER_FIELDS
        return {
            "query": q,
            "searched_fields": fields,
            "dedup": DUPLICATE_CAP,
            "count": len(all_rows),
            "results": all_rows,
        }


def _pretty(data: Dict[str, Any], pretty: bool) -> Response:
    indent = 2 if pretty else None
    return Response(
        content=json.dumps(data, indent=indent, ensure_ascii=False),
        media_type="application/json",
    )


# --- Pydantic Schemas ---
class QueryItem(BaseModel):
    field: Optional[str] = "name"
    value: str
    mode: Optional[str] = "contains"
    limit: Optional[int] = 10


class BatchRequest(BaseModel):
    queries: List[QueryItem]
    limit: int = Field(default=10, ge=1, le=100)


# --- API Routes ---
@app.get("/")
def root():
    return {
        "app": "GP Data Search API (Remote HTTP)",
        "data_repo": DATA_REPO_URL,
        "index_repo": INDEX_REPO_URL,
        "indexes_active": _INDEX_STATUS,
        "columns": SEARCH_FIELDS,
        "parallelism": PARALLELISM,
        "dedup": DUPLICATE_CAP,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexes_active": _INDEX_STATUS,
        "data_sources": PARQUET_FILES,
    }


@app.get("/search")
async def search(
    q: str = Query(..., description="Search term - matches across remote dataset"),
    field: Optional[str] = Query(None, description=f"Restrict field: {SEARCH_FIELDS}"),
    mode: str = Query("contains", pattern="^(exact|contains)$"),
    limit: int = Query(10, ge=1, le=1000),
    source: Optional[str] = Query(None, pattern="^(icmr|hitek|inddata)$"),
    pretty: bool = Query(True, description="Pretty print JSON output"),
):
    """Executes range-request search directly over remote Hugging Face parquet data."""
    if field:
        try:
            data = _run_field_search(field, q, mode, limit, source)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        data = {
            "query": q,
            "field": field,
            "mode": mode,
            "dedup": DUPLICATE_CAP,
            "count": data["count"],
            "results": data["results"],
        }
    else:
        data = await _unified_search(q, limit, source)
    return _pretty(data, pretty)


@app.post("/search/parallel")
async def search_parallel(req: BatchRequest):
    """Runs up to 50 concurrent queries across DuckDB connection workers."""
    if not req.queries:
        raise HTTPException(status_code=400, detail="queries list must not be empty")
    if len(req.queries) > 50:
        raise HTTPException(status_code=400, detail="max 50 queries per batch")

    query_tuples = [
        (q.field or "name", q.value, q.mode or "contains", q.limit or req.limit)
        for q in req.queries
    ]

    results = await _parallel(query_tuples)
    return _pretty(
        {"searches": len(req.queries), "parallelism": PARALLELISM, "results": results},
        True,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)