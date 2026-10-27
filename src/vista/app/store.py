"""SQLite persistence for analysis history. Reports are stored as JSON; images live under data/runs/<id>/."""
import json
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

from vista.app import config

_lock = threading.Lock()
SCHEMA = """CREATE TABLE IF NOT EXISTS analyses(
 id TEXT PRIMARY KEY, created_at TEXT, image_id TEXT, filename TEXT, stream TEXT, category TEXT,
 stream_conf REAL, fine_conf REAL, det_conf REAL, uncertain INTEGER, contamination REAL, condition_score REAL,
 co2e REAL, value_usd REAL, latency_ms REAL, report TEXT)"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def db():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        con = sqlite3.connect(config.DB_PATH)
        con.row_factory = sqlite3.Row
        con.execute(SCHEMA)
        try:
            yield con
            con.commit()
        finally:
            con.close()


def save(report: dict) -> None:
    c = report["classification"]
    imp = report["carbon_economic"]
    with db() as con:
        con.execute("INSERT OR REPLACE INTO analyses VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            report["report_id"], report["timestamp_utc"], report["input"]["image_id"], report["input"]["filename"],
            c["stream"]["prediction"], c["fine"]["prediction"], c["stream"]["confidence"], c["fine"]["confidence"],
            report["detection"]["confidence"], int(c["uncertain"]), report["contamination"].get("score"),
            report["condition"].get("score"), imp.get("co2e_avoided_kg"), imp.get("recovery_value_usd"),
            report["timings"]["total_ms"], json.dumps(report)))


def get(aid: str) -> dict | None:
    with db() as con:
        r = con.execute("SELECT report FROM analyses WHERE id=?", (aid,)).fetchone()
    return json.loads(r["report"]) if r else None


def query(stream=None, category=None, min_conf=None, max_conf=None, date_from=None, date_to=None, limit=500) -> list[dict]:
    sql, args = "SELECT * FROM analyses WHERE 1=1", []
    for col, val, op in (("stream", stream, "="), ("category", category, "="), ("fine_conf", min_conf, ">="),
                         ("fine_conf", max_conf, "<="), ("created_at", date_from, ">="), ("created_at", date_to, "<=")):
        if val not in (None, ""):
            sql += f" AND {col} {op} ?"
            args.append(val)
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    with db() as con:
        return [{k: r[k] for k in r.keys() if k != "report"} for r in con.execute(sql, args).fetchall()]


def clear() -> None:
    with db() as con:
        con.execute("DELETE FROM analyses")
    for d in (config.RUN_DIR, config.UPLOAD_DIR):
        if d.exists():
            shutil.rmtree(d)
