import sqlite3

import pandas as pd

from quakes.etl import load_sqlite


def _count(db):
    return sqlite3.connect(db).execute("select count(*) from quakes").fetchone()[0]


def test_load_is_idempotent(tmp_path):
    db = tmp_path / "q.db"
    df = pd.DataFrame({"id": ["a", "b"], "mag": [4.1, 5.0]})
    load_sqlite(df, db, "quakes")
    load_sqlite(df, db, "quakes")
    assert _count(db) == 2


def test_load_updates_existing_row(tmp_path):
    db = tmp_path / "q.db"
    load_sqlite(pd.DataFrame({"id": ["a", "b"], "mag": [4.1, 5.0]}), db, "quakes")
    load_sqlite(pd.DataFrame({"id": ["a"], "mag": [9.9]}), db, "quakes")
    con = sqlite3.connect(db)
    assert con.execute("select mag from quakes where id='a'").fetchone()[0] == 9.9
    assert _count(db) == 2
