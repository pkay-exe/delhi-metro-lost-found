"""Generate demo data, train and evaluate the three classifiers, tune score weights, save everything.

Run:  python train.py                 (strict data, the default)
      python train.py --mode easy     (the first, easier version of the data, for comparison)
Outputs: data/*.csv, data/model.joblib, data/metrics.json, data/config.json, static/img/*.png
"""
import argparse
import json
import os
import random
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import data_gen as dg
import matcher as mt

warnings.filterwarnings("ignore")
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA, IMG = os.path.join(HERE, "data"), os.path.join(HERE, "static", "img")
os.makedirs(DATA, exist_ok=True)
os.makedirs(IMG, exist_ok=True)
FEATURE_COLS = ["lost_description", "found_description", "lost_category", "found_category",
                "lost_station", "found_station", "lost_date", "found_date"]
EVAL_BASE = pd.Timestamp("2026-10-01")  # "today" for the offline search test
REAL_WORLD_MATCH_RATE = 1 / 21          # one true match for every 20 non-matches

COLORS = {"Logistic Regression": "#2a78d6", "SVM": "#eb6834", "Random Forest": "#1baf7a"}
INK, SOFT, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": SOFT,
                     "xtick.color": SOFT, "ytick.color": SOFT, "axes.titlecolor": INK, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})


def build_models():
    clfs = {
        "Logistic Regression": LogisticRegression(C=10, max_iter=3000, random_state=SEED),
        "SVM": SVC(kernel="rbf", C=3, probability=True, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    }
    return {n: Pipeline([("features", mt.PairFeaturizer()), ("clf", c)]) for n, c in clfs.items()}


def precision_at_real_rate(cm, pi=REAL_WORLD_MATCH_RATE):
    """Precision if only 1 pair in 21 were a real match (the test set is 50/50), from the confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)
    return float(pi * tpr / (pi * tpr + (1 - pi) * fpr))


def main(mode):
    strict = mode == "strict"
    rng = random.Random(SEED)
    if strict:
        train_df = dg.make_training_data_strict(rng, 2000)
        found_df, hidden_items = dg.make_found_database_strict(rng, 500, exclude_groups=set(train_df["item_group"]))
    else:
        train_df = dg.make_training_data(rng, 2000)
        found_df, hidden_items = dg.make_found_database(rng, 500)
    train_df.to_csv(f"{DATA}/training_data.csv", index=False)
    found_df.to_csv(f"{DATA}/found_items.csv", index=False)
    print(f"Mode: {mode}. Training pairs {len(train_df)} (match rate {train_df['match'].mean():.2f}), "
          f"found items {len(found_df)}")

    # ---- split: strict mode keeps every item combination on one side only (no near-duplicate leakage) ----
    if strict:
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(train_df, groups=train_df["item_group"]))
        X_train, X_test = train_df.loc[tr, FEATURE_COLS], train_df.loc[te, FEATURE_COLS]
        y_train, y_test = train_df.loc[tr, "match"], train_df.loc[te, "match"]
        groups_train = train_df.loc[tr, "item_group"]
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            train_df[FEATURE_COLS], train_df["match"], test_size=0.2, stratify=train_df["match"], random_state=SEED)
        groups_train = None
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ---- five-fold cross-validation and hold-out test ----
    models, cv_res, test_res, cms, prev_prec = build_models(), {}, {}, {}, {}
    for name, pipe in models.items():
        r = cross_validate(pipe, X_train, y_train, cv=cv, groups=groups_train,
                           scoring=["accuracy", "precision", "recall", "f1"])
        cv_res[name] = {m: float(r[f"test_{m}"].mean()) for m in ["accuracy", "precision", "recall", "f1"]}
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        test_res[name] = {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
                          "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred)}
        cms[name] = confusion_matrix(y_test, pred)
        prev_prec[name] = precision_at_real_rate(cms[name])
        print(f"{name:20} CV acc {cv_res[name]['accuracy']:.3f} | test acc {test_res[name]['accuracy']:.3f} "
              f"prec {test_res[name]['precision']:.3f} rec {test_res[name]['recall']:.3f} "
              f"| precision at 1 match in 21: {prev_prec[name]:.2f}")

    # ---- search quality: does the true item rank high among 500 found items? ----
    f_eval = found_df.copy()
    f_eval["date_found"] = EVAL_BASE - pd.to_timedelta(f_eval["days_ago"], unit="D")
    qrng = random.Random(SEED + 1)
    idx = qrng.sample(range(len(found_df)), 500)
    make_q = dg.make_query_strict if strict else dg.make_query_for
    queries = [(i, make_q(qrng, hidden_items[i], found_df.iloc[i], EVAL_BASE)) for i in idx]
    val_q, test_q = queries[:200], queries[200:]

    def comps_for(pipe, qs):
        return [(i, mt.components(pipe, f_eval, q)) for i, q in qs]

    def rank_metrics(comp_list, score_fn):
        top1 = top5 = 0
        for i, comp in comp_list:
            order = np.argsort(-score_fn(comp))[:5]
            top1 += int(order[0] == i)
            top5 += int(i in order)
        return top1 / len(comp_list), top5 / len(comp_list)

    lr_val = comps_for(models["Logistic Regression"], val_q)
    grid = {"model only": {"model": 1, "text": 0, "location": 0, "time": 0},
            "default": mt.DEFAULT_WEIGHTS,
            "text heavy": {"model": .45, "text": .35, "location": .12, "time": .08},
            "location heavy": {"model": .45, "text": .2, "location": .25, "time": .10},
            "balanced": {"model": .5, "text": .3, "location": .1, "time": .1},
            "model heavy": {"model": .7, "text": .15, "location": .10, "time": .05}}
    val_scores = {k: rank_metrics(lr_val, lambda c, w=w: mt.blend(c, w)) for k, w in grid.items()}
    best = max(val_scores, key=lambda k: (val_scores[k][0], val_scores[k][1]))
    # One-standard-error rule: with only 200 validation searches, a gain smaller than the noise is not a real
    # gain, so keep the default weights unless another option beats them by more than one standard error.
    se = (val_scores[best][0] * (1 - val_scores[best][0]) / len(val_q)) ** 0.5
    if best != "default" and val_scores["default"][0] >= val_scores[best][0] - se:
        print(f"  best option '{best}' beats default by {val_scores[best][0] - val_scores['default'][0]:.3f}, "
              f"within one standard error ({se:.3f}); keeping default weights")
        best = "default"
    weights = grid[best]
    print("Weight search on 200 validation queries (top-1, top-5):")
    for k, v in val_scores.items():
        print(f"  {k:15} {v[0]:.2f} {v[1]:.2f}" + ("   <- chosen" if k == best else ""))

    retrieval = {}
    for name, pipe in models.items():
        tc = comps_for(pipe, test_q)
        p_only = rank_metrics(tc, lambda c: c["P"])
        blended = rank_metrics(tc, lambda c: mt.blend(c, weights))
        retrieval[name] = {"model_only_top1": p_only[0], "model_only_top5": p_only[1],
                           "blend_top1": blended[0], "blend_top5": blended[1]}
        print(f"{name:20} search test (300 queries): model only {p_only[0]:.2f}/{p_only[1]:.2f} | "
              f"blended {blended[0]:.2f}/{blended[1]:.2f}")

    # ---- score distribution for the likelihood labels ----
    true_scores, other_scores = [], []
    for i, comp in comps_for(models["Logistic Regression"], test_q):
        s = mt.blend(comp, weights)
        true_scores.append(s[i])
        other_scores.extend(np.delete(s, i)[:50])
    score_stats = {"true_item_median": float(np.median(true_scores)),
                   "true_item_share_ge_0.70": float(np.mean(np.array(true_scores) >= .70)),
                   "true_item_share_ge_0.50": float(np.mean(np.array(true_scores) >= .50)),
                   "other_items_share_ge_0.70": float(np.mean(np.array(other_scores) >= .70)),
                   "other_items_share_ge_0.50": float(np.mean(np.array(other_scores) >= .50))}
    print("Score stats:", {k: round(v, 3) for k, v in score_stats.items()})

    # ---- charts ----
    metrics = ["accuracy", "precision", "recall", "f1"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, res, title in [(axes[0], cv_res, "Five-fold cross-validation (mean)"),
                           (axes[1], test_res, "Hold-out test set")]:
        x = np.arange(4)
        for k, (name, color) in enumerate(COLORS.items()):
            vals = [res[name][m] for m in metrics]
            bars = ax.bar(x + (k - 1) * 0.28, vals, 0.26, color=color, edgecolor="#fcfcfb", linewidth=1.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + .004, f"{v:.2f}", ha="center", fontsize=8, color=SOFT)
        ax.set_xticks(x)
        ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-score"])
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="x", visible=False)
    lo = min(min(v[m] for v in list(cv_res.values()) + list(test_res.values()) for m in metrics), .9)
    axes[0].set_ylim(max(0, lo - .1), 1.04)
    axes[0].set_ylabel("Score (axis does not start at 0)")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()], list(COLORS), loc="upper center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(f"{IMG}/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette("#2a78d6", as_cmap=True), cbar=False,
                    linewidths=2, linecolor="#fcfcfb", ax=ax, annot_kws={"fontsize": 13})
        ax.set_title(name, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["Non-match", "Match"])
        ax.set_yticklabels(["Non-match", "Match"], rotation=0)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(f"{IMG}/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- final model for the website: Logistic Regression refit on all 2,000 pairs ----
    final = build_models()["Logistic Regression"].fit(train_df[FEATURE_COLS], train_df["match"])
    joblib.dump(final, f"{DATA}/model.joblib")
    json.dump({"weights": weights, "weights_name": best, "model": "Logistic Regression", "mode": mode},
              open(f"{DATA}/config.json", "w"), indent=1)
    json.dump({"mode": mode, "cross_validation": cv_res, "test": test_res, "retrieval": retrieval,
               "precision_at_1_in_21": prev_prec, "settings": dg.STRICT if strict else "easy",
               "weight_search": {k: {"top1": v[0], "top5": v[1]} for k, v in val_scores.items()},
               "score_stats": score_stats, "n_train": len(train_df), "n_found": len(found_df)},
              open(f"{DATA}/metrics.json", "w"), indent=1)
    print("Saved model, config, metrics and charts.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["strict", "easy"], default="strict")
    main(ap.parse_args().mode)
