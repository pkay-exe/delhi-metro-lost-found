"""Matching engine: text cleaning, pair features, scoring and explanations."""
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import stations as st

UNKNOWN = "Not sure"
# Illustrative starting weights; train.py picks the final ones on validation queries.
DEFAULT_WEIGHTS = {"model": 0.55, "text": 0.25, "location": 0.12, "time": 0.08}


# ---- text ------------------------------------------------------------------------------------------------
def clean_text(text):
    """Lowercase, replace special characters with spaces, collapse extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def mask_sensitive(text):
    """Hide long digit runs (ID, phone, card numbers) and e-mail addresses before anything is stored."""
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email hidden]", str(text))
    return re.sub(r"\d[\d\s-]{4,}\d", "[number hidden]", text)


COLOR_FAMILY = {"black": "black", "blue": "blue", "navy": "blue", "red": "red", "maroon": "red",
                "green": "green", "brown": "brown", "white": "white", "grey": "grey", "gray": "grey",
                "silver": "silver", "metallic": "silver", "pink": "pink", "yellow": "yellow"}


def color_set(cleaned_text):
    return {COLOR_FAMILY[w] for w in cleaned_text.split() if w in COLOR_FAMILY}


# ---- pair features ---------------------------------------------------------------------------------------
class PairFeaturizer(BaseEstimator, TransformerMixin):
    """Turns (lost report, found report) pairs into numbers.

    Text:      cosine similarity, word overlap, colour overlap/conflict, and the element-wise product of the
               two TF-IDF vectors (which weighted words both reports share).
    Category:  both chose the same category, or both chose different ones.
    Location:  same station, and closeness on the metro network (stops between the two stations).
    Time:      whether the item was found after it was lost, and how many days later.
    The TF-IDF vocabulary is learned inside fit(), so cross-validation never peeks at the held-out fold.
    """

    def __init__(self, max_features=500):
        self.max_features = max_features

    def fit(self, X, y=None):
        corpus = pd.concat([X["lost_description"], X["found_description"]]).map(clean_text)
        self.tfidf_ = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2),
                                      stop_words="english", sublinear_tf=True)
        self.tfidf_.fit(corpus)
        return self

    def transform(self, X):
        lost = X["lost_description"].map(clean_text)
        found = X["found_description"].map(clean_text)
        L, F = self.tfidf_.transform(lost), self.tfidf_.transform(found)
        shared = L.multiply(F).tocsr()
        cos = np.asarray(shared.sum(axis=1)).ravel()  # rows are L2-normalised, so this is the cosine
        jac = np.array([len(set(a.split()) & set(b.split())) / max(len(set(a.split()) | set(b.split())), 1)
                        for a, b in zip(lost, found)])

        cl, cf = [color_set(t) for t in lost], [color_set(t) for t in found]
        color_overlap = np.array([float(bool(a & b)) for a, b in zip(cl, cf)])
        color_conflict = np.array([float(bool(a) and bool(b) and not (a & b)) for a, b in zip(cl, cf)])

        lc, fc = X["lost_category"].values, X["found_category"].values
        known = np.array([(a != UNKNOWN) and (b != UNKNOWN) for a, b in zip(lc, fc)])
        same = np.array([a == b for a, b in zip(lc, fc)])
        cat_match, cat_conflict = (known & same).astype(float), (known & ~same).astype(float)

        h = np.array([st.hops(a, b) for a, b in zip(X["lost_station"], X["found_station"])], dtype=float)
        same_station = (h == 0).astype(float)
        prox = np.clip(1 - h / 8, 0, 1)

        gap = (pd.to_datetime(X["found_date"]) - pd.to_datetime(X["lost_date"])).dt.days.values.astype(float)
        valid = (gap >= -1).astype(float)  # found before it was lost is not possible (1 day of slack)
        recency = valid * np.exp(-np.clip(gap, 0, None) / 7)

        dense = np.c_[cos, jac, color_overlap, color_conflict, cat_match, cat_conflict,
                      same_station, prox, h / st.MAX_HOPS, valid, recency, np.clip(gap, -1, 14) / 14]
        return sparse.hstack([sparse.csr_matrix(dense), shared]).tocsr()


# ---- searching -------------------------------------------------------------------------------------------
def components(pipe, found_df, q):
    """Per-candidate signals for one lost report: P (model), C (text), L (location), T (time), plus hops/gap."""
    cand = found_df.reset_index(drop=True)
    pairs = pd.DataFrame({
        "lost_description": q["description"], "lost_category": q["category"],
        "lost_station": q["station"], "lost_date": q["date"],
        "found_description": cand["description"].values, "found_category": cand["category"].values,
        "found_station": cand["station"].values, "found_date": cand["date_found"].values})
    P = pipe.predict_proba(pairs)[:, 1]
    feat = pipe.named_steps["features"]
    q_vec = feat.tfidf_.transform([clean_text(q["description"])])
    C = cosine_similarity(q_vec, feat.tfidf_.transform(cand["description"].map(clean_text))).ravel()
    h = np.array([st.hops(q["station"], s) for s in cand["station"]])
    gap = (pd.to_datetime(cand["date_found"]) - pd.to_datetime(q["date"])).dt.days.values
    L = np.clip(1 - h / 8, 0, 1)
    T = np.where(gap >= -1, np.exp(-np.clip(gap, 0, None) / 7), 0.0)
    return {"P": P, "C": C, "L": L, "T": T, "hops": h, "gap": gap}


def blend(comp, w):
    return w["model"] * comp["P"] + w["text"] * comp["C"] + w["location"] * comp["L"] + w["time"] * comp["T"]


def find_matches(pipe, found_df, q, weights=None, top_k=5):
    """Rank found items for one lost report. Returns the top_k rows with all signals and a final score."""
    w = weights or DEFAULT_WEIGHTS
    if found_df.empty:
        return pd.DataFrame()
    comp = components(pipe, found_df, q)
    cand = found_df.reset_index(drop=True).copy()
    cand["probability"] = comp["P"].round(3)
    cand["text_similarity"] = comp["C"].round(3)
    cand["hops"] = comp["hops"]
    cand["gap_days"] = comp["gap"]
    cand["score"] = blend(comp, w).round(3)
    return cand.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)


def likelihood(score, category_conflict=False):
    """Plain-language label for a ranking score. These cut-offs are heuristics, not calibrated probabilities.
    Two reports that name different categories are never better than a weak match."""
    if category_conflict:
        return "Weak match"
    if score >= 0.70:
        return "Strong match"
    if score >= 0.50:
        return "Possible match"
    return "Weak match"


_GENERIC = set(ENGLISH_STOP_WORDS) | {
    "found", "lost", "missing", "please", "help", "picked", "left", "someone", "containing", "having",
    "metro", "train", "seat", "platform", "gate", "coach", "near", "urgent", "hidden", "number", "email"}


def explain(q, row):
    """Short reasons behind a match, in plain words."""
    ql, fl = clean_text(q["description"]), clean_text(row["description"])
    reasons = []
    shared = [w for w in dict.fromkeys(ql.split()) if w in set(fl.split()) and w not in _GENERIC and len(w) > 2]
    if shared:
        reasons.append("both mention " + ", ".join(shared[:4]))
    colors = color_set(ql) & color_set(fl)
    if colors:
        reasons.append("same colour (" + ", ".join(sorted(colors)) + ")")
    elif color_set(ql) and color_set(fl):
        reasons.append("different colours reported")
    if q["category"] != UNKNOWN and row["category"] != UNKNOWN:
        reasons.append("same category" if q["category"] == row["category"] else "different category")
    h = int(row["hops"])
    reasons.append("handed in at the same station" if h == 0 else
                   f"handed in {h} stop{'s' if h != 1 else ''} away" if h < st.MAX_HOPS else
                   "handed in far from where you lost it")
    g = int(row["gap_days"])
    if g < -1:
        reasons.append("found before the date you lost it")
    elif g <= 0:
        reasons.append("found the same day")
    else:
        reasons.append(f"found {g} day{'s' if g != 1 else ''} later")
    return reasons
