import json
import os
import sqlite3
from typing import Any, Dict, Optional

_DB_PATH = os.path.join("/workspace", "data", "app.sqlite3")


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


_CONN = _get_conn()


def init_db() -> None:
    _CONN.execute(
        """
        CREATE TABLE IF NOT EXISTS question_papers (
            id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            title TEXT,
            data_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    _CONN.execute(
        """
        CREATE TABLE IF NOT EXISTS attempts (
            id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL,
            answers_json TEXT NOT NULL,
            score INTEGER NOT NULL,
            total INTEGER NOT NULL,
            breakdown_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    _CONN.commit()


def save_paper(paper_id: str, dataset_id: str, title: str, paper: Dict[str, Any]) -> None:
    _CONN.execute(
        "INSERT OR REPLACE INTO question_papers (id, dataset_id, title, data_json) VALUES (?, ?, ?, ?)",
        (paper_id, dataset_id, title, json.dumps(paper)),
    )
    _CONN.commit()


def get_paper(paper_id: str) -> Optional[Dict[str, Any]]:
    cur = _CONN.execute(
        "SELECT data_json FROM question_papers WHERE id = ? LIMIT 1",
        (paper_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return json.loads(row[0])


def save_attempt(attempt_id: str, paper_id: str, answers: Dict[str, Any], score: int, total: int, breakdown: Any) -> None:
    _CONN.execute(
        "INSERT OR REPLACE INTO attempts (id, paper_id, answers_json, score, total, breakdown_json) VALUES (?, ?, ?, ?, ?, ?)",
        (attempt_id, paper_id, json.dumps(answers), score, total, json.dumps(breakdown)),
    )
    _CONN.commit()