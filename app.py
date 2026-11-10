"""Delhi Metro lost-and-found prototype: a small Flask website around the matching model.

Run:  python train.py      (once, creates the model)
      python app.py        (then open http://127.0.0.1:5000)
"""
import json
import os
from datetime import date, timedelta

import joblib
import pandas as pd
from flask import Flask, abort, redirect, render_template, request, url_for

import db
import matcher as mt
import stations as st
from data_gen import CATEGORIES

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

MODEL_PATH = os.path.join(HERE, "data", "model.joblib")
if not os.path.exists(MODEL_PATH):
    raise SystemExit("No trained model found. Run  python train.py  first.")
MODEL = joblib.load(MODEL_PATH)
CONFIG = json.load(open(os.path.join(HERE, "data", "config.json")))
METRICS = json.load(open(os.path.join(HERE, "data", "metrics.json")))
WEIGHTS = CONFIG["weights"]
db.init_db()

CATEGORY_CHOICES = [mt.UNKNOWN] + CATEGORIES
STATION_LINES = {s: [{"name": l, "color": st.LINES[l]["color"], "ink": st.LINES[l]["ink"]} for l in st.lines_of(s)]
                 for s in st.STATIONS}


def line_pills(station):
    return STATION_LINES.get(station, [])


app.jinja_env.globals.update(line_pills=line_pills, likelihood=mt.likelihood, lines=st.LINES)


def route_strip(a, b):
    """Render plan for the route between two stations: stops, coloured segments, and a gap if it is long."""
    p = st.path(a, b)
    n = len(p)
    seg_color = lambda i: st.LINES[st.edge_line(p[i], p[i + 1])]["color"] if st.edge_line(p[i], p[i + 1]) else "#9aa5b5"
    items = []
    if n == 1:
        return {"parts": [{"t": "stop", "end": True, "name": p[0]}], "stops": p}
    keep = list(range(n)) if n <= 9 else [0, 1, 2, n - 3, n - 2, n - 1]
    for k, i in enumerate(keep):
        items.append({"t": "stop", "end": i in (0, n - 1), "name": p[i]})
        if k < len(keep) - 1:
            if keep[k + 1] != i + 1:
                items.append({"t": "gap", "hidden": keep[k + 1] - i - 1})
            else:
                items.append({"t": "seg", "color": seg_color(i)})
    return {"parts": items, "stops": p}


def lines_on_path(p):
    seen = []
    for i in range(len(p) - 1):
        ln = st.edge_line(p[i], p[i + 1])
        if ln and ln not in seen:
            seen.append(ln)
    return [{"name": l, "color": st.LINES[l]["color"], "ink": st.LINES[l]["ink"]} for l in seen]


# ---- helpers -----------------------------------------------------------------------------------------------
def parse_report(form, kind):
    """Validate a lost/found form. Returns (clean_values, errors)."""
    desc = mt.mask_sensitive(" ".join(form.get("description", "").split()))
    category = form.get("category", mt.UNKNOWN)
    station = st.normalize_station(form.get("station", ""))
    when = form.get("date", "")
    errors = {}
    if len(desc) < 8:
        errors["description"] = "Describe the item in a few words, for example colour, type and anything on it."
    if len(desc) > 300:
        errors["description"] = "Keep the description under 300 characters."
    if category not in CATEGORY_CHOICES:
        category = mt.UNKNOWN
    if station is None:
        errors["station"] = "Choose a station from the list. Start typing and pick one of the suggestions."
    try:
        d = date.fromisoformat(when)
        if d > date.today():
            errors["date"] = "The date cannot be in the future."
    except ValueError:
        errors["date"] = "Pick a date."
    return {"description": desc, "category": category, "station": station, "date": when}, errors


def run_search(values, top_k=6):
    found = db.found_df()
    res = mt.find_matches(MODEL, found, values, WEIGHTS, top_k=top_k)
    matches = []
    for _, r in res.iterrows():
        row = r.to_dict()
        row["reasons"] = mt.explain(values, row)
        conflict = (values["category"] != mt.UNKNOWN and row["category"] != mt.UNKNOWN
                    and values["category"] != row["category"])
        row["label"] = mt.likelihood(row["score"], conflict)
        row["score10"] = max(1.0, round(float(row["score"]) * 10, 1))  # shown to people as x out of 10
        row["route"] = route_strip(values["station"], row["station"])
        row["route_lines"] = lines_on_path(row["route"]["stops"])
        row["found_on"] = pd.Timestamp(row["date_found"]).strftime("%d %b %Y")
        matches.append(row)
    return matches


# ---- pages -------------------------------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html", categories=CATEGORY_CHOICES, station_names=st.STATIONS,
                           station_lines=STATION_LINES, values={}, errors={}, today=date.today().isoformat(),
                           n_found=len(db.found_df()))


@app.post("/lost")
def lost():
    values, errors = parse_report(request.form, "lost")
    if errors:
        return render_template("home.html", categories=CATEGORY_CHOICES, station_names=st.STATIONS,
                               station_lines=STATION_LINES, values=request.form, errors=errors,
                               today=date.today().isoformat(), n_found=len(db.found_df())), 400
    ref = db.add_lost(values["description"], values["category"], values["station"], values["date"])
    return redirect(url_for("status", ref=ref))


@app.route("/status", methods=["GET", "POST"])
def status_lookup():
    if request.method == "POST":
        ref = request.form.get("ref", "").strip().upper()
        if db.get_lost(ref):
            return redirect(url_for("status", ref=ref))
        return render_template("status_lookup.html", error="We could not find that reference code. "
                               "Check it and try again.", ref=ref), 404
    return render_template("status_lookup.html", error=None, ref="")


@app.route("/status/<ref>")
def status(ref):
    rep = db.get_lost(ref)
    if not rep:
        abort(404)
    values = {"description": rep["description"], "category": rep["category"], "station": rep["station"],
              "date": rep["date_lost"]}
    matches = run_search(values)
    strong = sum(1 for m in matches if m["label"] == "Strong match")
    return render_template("results.html", rep=rep, values=values, matches=matches, strong=strong,
                           lost_on=pd.Timestamp(rep["date_lost"]).strftime("%d %b %Y"))


@app.route("/found", methods=["GET", "POST"])
def found():
    ctx = dict(categories=CATEGORY_CHOICES, station_names=st.STATIONS, station_lines=STATION_LINES,
               today=date.today().isoformat())
    if request.method == "GET":
        return render_template("found.html", values={}, errors={}, **ctx)
    values, errors = parse_report(request.form, "found")
    if errors:
        return render_template("found.html", values=request.form, errors=errors, **ctx), 400
    code = db.add_found(values["description"], values["category"], values["station"], values["date"])
    # how many open lost reports look similar? (counts only, nothing about the people)
    lost = db.lost_df()
    similar = 0
    if not lost.empty:
        pairs = pd.DataFrame({
            "lost_description": lost["description"], "lost_category": lost["category"],
            "lost_station": lost["station"], "lost_date": lost["date_lost"],
            "found_description": values["description"], "found_category": values["category"],
            "found_station": values["station"], "found_date": values["date"]})
        similar = int((MODEL.predict_proba(pairs)[:, 1] >= 0.7).sum())
    return render_template("found_done.html", code=code, values=values, similar=similar)


@app.route("/browse")
def browse():
    df = db.found_df().sort_values(["date_found", "item_id"], ascending=[False, False])
    line = request.args.get("line", "")
    category = request.args.get("category", "")
    q = request.args.get("q", "").strip().lower()
    if line in st.LINES:
        on_line = {s for s in st.STATIONS if line in st.lines_of(s)}
        df = df[df["station"].isin(on_line)]
    if category in CATEGORIES:
        df = df[df["category"] == category]
    if q:
        df = df[df["station"].str.lower().str.contains(q) | df["description"].str.lower().str.contains(q)]
    p = request.args.get("page", "1")
    page = int(p) if p.isdigit() and int(p) > 0 else 1
    per = 20
    total = len(df)
    rows = df.iloc[(page - 1) * per: page * per].to_dict("records")
    for r in rows:
        r["found_on"] = pd.Timestamp(r["date_found"]).strftime("%d %b")
    args = {k: v for k, v in request.args.items() if k != "page"}
    return render_template("browse.html", rows=rows, total=total, page=page, pages=max(1, -(-total // per)),
                           line=line, category=category, q=q, categories=CATEGORIES, args=args)


@app.route("/about")
def about():
    return render_template("about.html", m=METRICS, cfg=CONFIG, n_stations=len(st.STATIONS),
                           n_interchange=sum(st.is_interchange(s) for s in st.STATIONS))


@app.errorhandler(404)
def not_found(_):
    return render_template("message.html", title="Page not found",
                           body="That page does not exist. Use the menu to go back."), 404


if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)))
