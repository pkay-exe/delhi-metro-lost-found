# Metro Lost & Found: Delhi Metro prototype

A web prototype that matches lost-item reports with found-item reports on the Delhi Metro network, using
machine learning. It is a student project running on demo data and is not connected to DMRC.

## Run it

```bash
pip install -r requirements.txt
python train.py        # about 2 minutes: generates data, trains 3 models, saves the model and charts
python app.py          # then open http://127.0.0.1:5000
```

Click "Fill an example" on the home page for a search that has matches in the demo data.

## What is inside

| File | Purpose |
|---|---|
| `stations.py` | Metro network: 8 lines, 213 stations, interchanges, stops-between-stations, shortest route |
| `data_gen.py` | Synthetic labelled pairs and found items (replace with real data when you have it) |
| `matcher.py` | Text cleaning, privacy masking, pair features, scoring, plain-language explanations |
| `train.py` | Cross-validates Logistic Regression, SVM and Random Forest, tunes score weights, saves the model |
| `db.py` | SQLite storage; `python db.py` resets the demo data with fresh dates |
| `app.py`, `templates/`, `static/` | The Flask website |

## How matching works

Each (lost, found) pair is turned into features: TF-IDF cosine similarity, word overlap, colour agreement or
conflict, category agreement, same station, closeness in stops, and whether the item was found after it was lost.
A classifier gives the probability that the pair is a match. Candidates are then ranked by

`S = 0.45·P + 0.20·C + 0.25·L + 0.10·T`

(P model probability, C text similarity, L closeness in stops, T timing). The website shows S × 10 as a score out of 10 (0.78 appears as 7.8). The weights were chosen on 200 validation
searches, and the blended options landed within about three points of each other.

## Using your own data

Replace `data/training_data.csv` (columns: `lost_description, found_description, lost_category, found_category,
lost_station, found_station, lost_date, found_date, match`; the generated file also has `pair_id` and `item_group`,
which are only used to keep identical items out of both the training and test sets) and `data/found_items.csv` (`item_id, description,
category, station, days_ago`), then re-run `train.py`. Station names must match `stations.py`.

## The data: strict version

`train.py` generates a deliberately hard synthetic dataset (`python train.py --mode strict`, the default):
- 2,000 pairs, exactly 1,000 matches and 1,000 non-matches; no missing values and no duplicate pairs.
- Look-alike non-matches (same category, often same colour), pairs of near-identical items, vague reports, colour
  mix-ups (black vs navy), wrong or unknown categories, typos and special characters.
- No date shortcut: matches and non-matches have the same date pattern. Matches are still closer in place.
- Train and test are split by item, so no exact item combination appears on both sides; found items used in the
  search test never appear in the training data.
- Expected accuracy: about 85-86% in cross-validation and 87-88% on the held-out pairs.

The first, easier version of the data and its results are kept in `data/easy_version/` for comparison
(`python train.py --mode easy` recreates it). It reached about 95-97% accuracy because of shortcuts in the data.

## Honest limits

- The data is synthetic. The strict version removes the obvious shortcuts, but real reports will still behave
  differently. Say this in your report and, if you can, test on a small set of real examples.
- At a realistic 1 match in 21 pairs, classification alone is not precise (about 20% of flagged pairs would be real
  matches), which is why the site ranks candidates for a person to check.
- The station list is a subset typed from public line maps. Check it against the official map.
- No contact details are collected. Ownership is verified at the station desk. Numbers and e-mail addresses in
  descriptions are masked before saving.
- There is no login, CSRF protection or rate limiting. Add them before any public deployment.
- The page loads the Atkinson Hyperlegible font from Google Fonts and falls back to system fonts offline.
