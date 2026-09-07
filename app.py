"""Gradio UI + FastAPI backend for ICMR + HITEK Search API.
Runs on Hugging Face Spaces (free Gradio tier).
Uses remote HF split indexes — no local database download needed."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from queue import Queue
from typing import Any, Dict, List, Optional

import duckdb
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from pydantic import BaseModel, Field

# ── Config ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

HF_INDEX_BASE = os.environ.get(
    "ICMR_HF_INDEX_BASE",
    "https://huggingface.co/datasets/HiTeckGroup/HiTeckgroop/resolve/main",
).rstrip("/")

INDEX_SOURCE = os.environ.get("ICMR_INDEX_SOURCE", "remote").lower()
PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "2"))
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

REMOTE_INDEXES = {
    "phone": [f"{HF_INDEX_BASE}/idx_phone.{i}.parquet" for i in range(7)],
    "aadhar": [f"{HF_INDEX_BASE}/idx_aadhar.{i}.parquet" for i in range(7)],
}

# ── Thread-Safe DuckDB Connection Pool ────────────────────────────────────────
_connection_pool: Queue[duckdb.DuckDBPyConnection] = Queue(maxsize=PARALLELISM)
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")


def _idx_ready(kind: str) -> bool:
    return kind in REMOTE_INDEXES


def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    # DuckDB 1.0+ extension auto-loading and HTTP performance caching
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET enable_object_cache = true;")
    con.execute(f"SET threads = {THREADS_PER_CONN};")

    for kind, urls in REMOTE_INDEXES.items():
        view = f"people_{kind}"
        lst = ", ".join(f"'{u}'" for u in urls)
        con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet([{lst}])")
    return con


def _init_pool():
    for _ in range(PARALLELISM):
        _connection_pool.put(_new_conn())


_init_pool()

# ── Data Sanitization & Dedup ───────────────────────────────────────────────
FORBIDDEN_FIELDS = {"aadharNumber", "aadhar", "aadharnumber"}


def _sanitize_record(row: dict) -> dict:
    """Strict zero-disclosure redaction policy for sensitive identification numbers."""
    sanitized = {}
    for k, v in row.items():
        if k in FORBIDDEN_FIELDS:
            sanitized[k] = "[Aadhaar Redacted]"
        else:
            sanitized[k] = v
    return sanitized


def _person_key(row: dict) -> tuple:
    ph = str(row.get("phoneNumber") or "").strip()
    ad = str(row.get("aadharNumber") or "").strip()
    if ph or ad:
        return (ph, ad)
    return (str(row.get("name") or "").strip(), str(row.get("fathersName") or "").strip())


def _connected_numbers(row: dict) -> List[dict]:
    connected, seen = [], set()
    for field in NUMBER_FIELDS:
        raw = row.get(field)
        if raw is None:
            continue
        
        # Enforce redaction in connected numbers output
        if field in FORBIDDEN_FIELDS:
            value = "[Aadhaar Redacted]"
        else:
            value = str(raw).strip()
            
        if not value or value in seen:
            continue
        seen.add(value)
        connected.append({"field": field, "value": value})
    return connected


def _cap_duplicates(rows: List[dict]) -> List[dict]:
    seen: Dict[tuple, int] = {}
    out = []
    for r in rows:
        k = _person_key(r)
        n = seen.get(k, 0)
        if n < DUPLICATE_CAP:
            seen[k] = n + 1
            record = _sanitize_record(r)
            record["connected_numbers"] = _connected_numbers(r)
            out.append(record)
    return out

# ── Parameterized Search Logic ──────────────────────────────────────────────
def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")

    fetch_limit = limit * DUPLICATE_CAP + 20
    con = _connection_pool.get()

    try:
        if mode == "exact":
            if field == "phoneNumber" and _idx_ready("phone"):
                view = "people_phone"
            elif field == "aadharNumber" and _idx_ready("aadhar"):
                view = "people_aadhar"
            else:
                return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}

            sql = f"SELECT * FROM {view} WHERE {field} = ? LIMIT ?"
            rows = con.execute(sql, [value, fetch_limit]).fetchall()

        elif mode == "contains":
            if field == "name":
                return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
            pattern = f"%{value}%"
            sql = f"SELECT * FROM people_phone WHERE {field} ILIKE ? LIMIT ?"
            rows = con.execute(sql, [pattern, fetch_limit]).fetchall()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        cols = [d[0] for d in con.description]
        results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
        return {"field": field, "value": value, "mode": mode, "count": len(results), "results": results}
    finally:
        _connection_pool.put(con)


def _unified_search(q: str, limit: int = 10) -> dict:
    q = q.strip()
    is_num = q.isdigit() and len(q) >= 8

    if is_num:
        all_rows = []
        searched = []

        if _idx_ready("phone"):
            r = _run_field_search("phoneNumber", q, "exact", limit)
            all_rows.extend(r["results"])
            searched.append("phoneNumber")

        if not all_rows and _idx_ready("aadhar"):
            r = _run_field_search("aadharNumber", q, "exact", limit)
            all_rows.extend(r["results"])
            searched.append("aadharNumber")

        all_rows = _cap_duplicates(all_rows)[:limit]
        return {
            "query": q,
            "searched_fields": searched,
            "count": len(all_rows),
            "results": all_rows,
        }
    else:
        return {"query": q, "searched_fields": [], "count": 0, "results": []}

# ── FastAPI (API Endpoints) ─────────────────────────────────────────────────
fastapi_app = FastAPI(title="GP Db Search API", version="2.0")

fastapi_app = FastAPI(title="GP Db Search API", version="2.0")

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryItem(BaseModel):
    field: str = Field(default="phoneNumber")
    value: str = Field(default="")
    mode: str = Field(default="exact")
    limit: Optional[int] = None


class BatchRequest(BaseModel):
    queries: List[QueryItem]
    limit: int = Field(default=10, ge=1, le=100)


@fastapi_app.get("/")
def root():
    return {
        "app": "ICMR + HITEK Search API",
        "records": 2_504_793_870,
        "indexes": {
            "phone": _idx_ready("phone"),
            "aadhar": _idx_ready("aadhar"),
        },
        "index_source": INDEX_SOURCE,
        "columns": SEARCH_FIELDS,
        "docs": "/docs",
    }


@fastapi_app.get("/health")
def health():
    return {
        "status": "ok",
        "raw_database_required": False,
        "indexes": {
            "phone": _idx_ready("phone"),
            "aadhar": _idx_ready("aadhar"),
        },
        "index_source": INDEX_SOURCE,
    }


@fastapi_app.get("/search")
async def search(
    q: Optional[str] = Query(None),
    mobile: Optional[str] = Query(None),
    field: Optional[str] = Query(None),
    mode: str = Query("exact"),
    limit: int = Query(10, ge=1, le=1000),
    pretty: bool = Query(True),
):
    q_val = (q or mobile or "").strip()
    if not q_val:
        raise HTTPException(422, "Provide 'q' or 'mobile'")

    if field:
        data = await asyncio.to_thread(_run_field_search, field, q_val, mode, limit)
    else:
        data = await asyncio.to_thread(_unified_search, q_val, limit)

    result = {
        "success": bool(data["count"]),
        **data,
        "number": q_val,
        "total": data["count"],
    }

    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


@fastapi_app.post("/search/parallel")
async def search_parallel(req: BatchRequest):
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 50:
        raise HTTPException(400, "max 50 queries per batch")

    tasks = [
        asyncio.to_thread(
            _run_field_search,
            item.field,
            item.value,
            item.mode,
            item.limit or req.limit,
        )
        for item in req.queries
    ]

    results = await asyncio.gather(*tasks)

    return Response(
        content=json.dumps({"searches": len(req.queries), "results": list(results)}, indent=2, ensure_ascii=False),
        media_type="application/json",
    )

# ── Gradio UI ───────────────────────────────────────────────────────────────
def format_result(row: dict) -> str:
    lines = []
    for field in SEARCH_FIELDS:
        val = row.get(field, "")
        if val:
            lines.append(f"**{field}:** {val}")

    cn = row.get("connected_numbers", [])
    if cn:
        nums = ", ".join(f"{c['field']}={c['value']}" for c in cn)
        lines.append(f"**connected:** {nums}")

    return "\n\n".join(lines)


def search_ui(query: str, limit: int) -> str:
    if not query or not query.strip():
        return "⚠️ Kuch toh search karo — phone number daalo."

    q = query.strip()
    try:
        data = _unified_search(q, int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"

    count = data["count"]
    results = data["results"]
    searched = ", ".join(data.get("searched_fields", []))

    if not results:
        return f"🔍 **Query:** `{q}`\n**Searched:** {searched}\n\n❌ **No data found**."

    header = f"🔍 **Query:** `{q}` | **Found:** {count} results | **Searched:** {searched}\n\n---\n\n"
    parts = [f"### Result {i}\n{format_result(row)}" for i, row in enumerate(results, 1)]

    return header + "\n\n---\n\n".join(parts)


def build_ui():
    with gr.Blocks(
        title="ICMR Search API",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 0; }
        .subtitle { text-align: center; color: #666; margin-top: 0; }
        """,
    ) as demo:
        gr.Markdown("# 🔍 ICMR + HITEK Search API", elem_classes="main-title")
        gr.Markdown("Search **2.5 billion records** — phone, name, address & more", elem_classes="subtitle")

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Phone number, name, or parameter daalo...",
                    lines=1,
                )
            with gr.Column(scale=1):
                limit_slider = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1,
                    label="Max Results",
                )

        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
        output = gr.Markdown(label="Results")

        search_btn.click(
            fn=search_ui,
            inputs=[query_input, limit_slider],
            outputs=output,
        )
        query_input.submit(
            fn=search_ui,
            inputs=[query_input, limit_slider],
            outputs=output,
        )

        gr.Markdown("---")

        with gr.Accordion("📡 API Info", open=False):
            gr.Markdown("""
            **Endpoints** (via FastAPI):
            - `GET /search?q=<query>` — Phone/Query search
            - `GET /search?mobile=<number>` — Phone search (alias)
            - `GET /health` — Health check
            - `GET /docs` — Swagger UI
            
            **Source:** [HF Dataset](https://huggingface.co/datasets/HiTeckGroup/HiTeckgroop)
            """)

        return demo


# ── Mount Gradio on FastAPI ─────────────────────────────────────────────────
demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)