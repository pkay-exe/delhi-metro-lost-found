"""SQLite storage for found items and lost reports. No contact details are collected (privacy by design)."""
import os
import random
import sqlite3
import string
from datetime import date, datetime, timedelta

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "data", "lostfound.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS found_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT, item_code TEXT UNIQUE, description TEXT NOT NULL,
    category TEXT NOT NULL, station TEXT NOT NULL, date_found TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS lost_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ref TEXT UNIQUE, description TEXT NOT NULL,
    category TEXT NOT NULL, station TEXT NOT NULL, date_lost TEXT NOT NULL, created_at TEXT NOT NULL);
"""


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect() as c:
        c.executescript(SCHEMA)
        if c.execute("SELECT COUNT(*) FROM found_items").fetchone()[0] == 0:
            seed(c)


# A few hand-written showcase items so the "Fill an example" search on the home page has something to find.
DEMO_ITEMS = [
    ("Black leather purse with an ID card and some debit cards", "Wallet", "Green Park", 3),
    ("Black Fastrack wallet, bank cards and a zipper pocket", "Wallet", "Rajiv Chowk", 2),
    ("Brown wallet with a metro card and some cash", "Wallet", "Saket", 2),
    ("Blue backpack with a laptop inside and a water bottle", "Bag", "Kashmere Gate", 1),
]


def seed(conn=None):
    """Load the demo found items. Dates are set relative to today so the demo always looks current."""
    own = conn is None
    conn = conn or connect()
    csv = os.path.join(HERE, "data", "found_items.csv")
    df = pd.read_csv(csv)
    today = date.today()
    rows = [(r.item_id, r.description, r.category, r.station,
             (today - timedelta(days=int(r.days_ago))).isoformat(), datetime.now().isoformat(timespec="seconds"))
            for r in df.itertuples()]
    stamp = datetime.now().isoformat(timespec="seconds")
    rows += [(f"D{i + 1:03d}", desc, cat, stn, (today - timedelta(days=ago)).isoformat(), stamp)
             for i, (desc, cat, stn, ago) in enumerate(DEMO_ITEMS)]
    conn.executemany("INSERT OR IGNORE INTO found_items (item_code, description, category, station, date_found,"
                     " created_at) VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    if own:
        conn.close()


def reset():
    with connect() as c:
        c.executescript("DROP TABLE IF EXISTS found_items; DROP TABLE IF EXISTS lost_reports;")
    init_db()


def found_df():
    with connect() as c:
        df = pd.read_sql_query("SELECT item_code AS item_id, description, category, station, date_found "
                               "FROM found_items", c)
    df["date_found"] = pd.to_datetime(df["date_found"])
    return df


def lost_df():
    with connect() as c:
        return pd.read_sql_query("SELECT ref, description, category, station, date_lost FROM lost_reports", c)


def _code(prefix, n=6):
    alphabet = string.ascii_uppercase.replace("O", "").replace("I", "") + "23456789"
    return prefix + "".join(random.choices(alphabet, k=n))


def add_lost(description, category, station, date_lost):
    ref = _code("LR-")
    with connect() as c:
        c.execute("INSERT INTO lost_reports (ref, description, category, station, date_lost, created_at)"
                  " VALUES (?,?,?,?,?,?)", (ref, description, category, station, date_lost,
                                            datetime.now().isoformat(timespec="seconds")))
    return ref


def get_lost(ref):
    with connect() as c:
        return c.execute("SELECT * FROM lost_reports WHERE ref = ?", (ref.strip().upper(),)).fetchone()


def add_found(description, category, station, date_found):
    with connect() as c:
        cur = c.execute("INSERT INTO found_items (item_code, description, category, station, date_found,"
                        " created_at) VALUES (?,?,?,?,?,?)",
                        ("TMP", description, category, station, date_found,
                         datetime.now().isoformat(timespec="seconds")))
        code = f"F{cur.lastrowid + 1000:04d}"
        c.execute("UPDATE found_items SET item_code = ? WHERE id = ?", (code, cur.lastrowid))
    return code


if __name__ == "__main__":
    reset()
    print("Database reset and re-seeded with fresh dates.")
