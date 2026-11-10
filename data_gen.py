"""Synthetic training data for the Delhi Metro lost-and-found matcher.

Real lost-and-found records are not public, so this generates believable demonstration data: each row pairs a
lost report with a found report. A pair is a match (1) when both were written about the same hidden item.
Scores measured on this data are optimistic because the wording comes from templates.
"""
import random

import pandas as pd

import stations as st

COLORS = ["black", "blue", "red", "green", "brown", "white", "grey", "silver",
          "pink", "yellow", "navy", "maroon"]
COLOR_WORDS = {"grey": ["grey", "gray"], "navy": ["navy", "dark blue"],
               "silver": ["silver", "metallic"], "maroon": ["maroon", "dark red"]}

UNKNOWN = "Not sure"

CATEGORY_SPECS = {
    "Bag": {"nouns": ["bag", "backpack", "laptop bag", "sling bag", "handbag", "tote bag"],
            "brands": ["Wildcraft", "Skybags", "American Tourister", "Puma", "Safari"],
            "features": ["laptop inside", "notebooks", "tiffin box", "water bottle", "keychain on the zip",
                         "umbrella in the side pocket", "office files", "two zips"]},
    "Phone": {"nouns": ["phone", "mobile", "smartphone", "cell phone"],
              "brands": ["Samsung", "iPhone", "Redmi", "OnePlus", "Vivo", "Oppo", "Realme"],
              "features": ["cracked screen", "black cover", "transparent case", "wallet case",
                           "sticker on the back", "ring holder", "lock screen photo"]},
    "Wallet": {"nouns": ["wallet", "purse", "billfold", "card holder"],
               "brands": ["Hidesign", "Puma", "Fastrack", "Wildcraft"],
               "features": ["ID card", "debit cards", "metro card", "some cash", "driving licence",
                            "bank cards", "zipper pocket", "college ID"]},
    "ID or metro card": {"nouns": ["ID card", "metro card", "card holder", "student ID", "office ID"],
                         "brands": [],
                         "features": ["blue lanyard", "office logo", "photo ID", "plastic holder",
                                      "college name printed", "lanyard"]},
    "Earphones": {"nouns": ["earbuds", "earphones", "wireless earbuds", "headphones"],
                  "brands": ["boAt", "Realme", "Sony", "JBL", "Apple"],
                  "features": ["charging case", "wired cable", "black case", "scratched case",
                               "left earbud only"]},
    "Laptop or tablet": {"nouns": ["laptop", "tablet", "notebook computer"],
                         "brands": ["HP", "Dell", "Lenovo", "Asus", "Acer", "Apple"],
                         "features": ["sticker on the lid", "black sleeve", "charger", "scratches",
                                      "college name sticker"]},
    "Bottle or tiffin": {"nouns": ["water bottle", "tiffin box", "lunch box", "flask"],
                         "brands": ["Milton", "Cello", "Tupperware"],
                         "features": ["steel body", "insulated", "blue lid", "food inside", "two compartments"]},
    "Umbrella": {"nouns": ["umbrella", "folding umbrella"], "brands": [],
                 "features": ["wooden handle", "floral print", "automatic", "broken rib", "plastic cover"]},
    "Keys": {"nouns": ["keys", "key bunch", "keychain", "key ring"], "brands": [],
             "features": ["car key", "three keys", "red tag", "bike key", "house keys", "remote lock",
                          "small toy charm"]},
    "Glasses": {"nouns": ["glasses", "spectacles", "eyeglasses", "sunglasses"],
                "brands": ["Ray-Ban", "Titan", "Vincent Chase"],
                "features": ["black frame", "hard case", "round lens", "blue frame", "cloth pouch"]},
    "Clothing": {"nouns": ["jacket", "shawl", "sweater", "dupatta", "cap", "scarf"], "brands": [],
                 "features": ["woollen", "zip", "hood", "stripes", "label inside", "pocket"]},
    "Books or documents": {"nouns": ["file", "folder", "book", "notebook", "document folder"],
                           "brands": [],
                           "features": ["exam papers", "certificates", "plastic folder", "handwritten notes",
                                        "rubber band"]},
    "Watch or jewellery": {"nouns": ["watch", "wristwatch", "bracelet", "smartwatch"],
                           "brands": ["Titan", "Casio", "Fossil", "Noise", "Fastrack"],
                           "features": ["leather strap", "metal strap", "digital display", "scratched glass",
                                        "round dial"]},
    "Charger or power bank": {"nouns": ["power bank", "charger", "charging cable"],
                              "brands": ["Mi", "Anker", "Ambrane", "Realme"],
                              "features": ["10000 mAh", "white cable", "type C cable", "black adapter"]},
    "Luggage": {"nouns": ["suitcase", "trolley bag", "duffle bag"],
                "brands": ["American Tourister", "Safari", "VIP", "Skybags"],
                "features": ["hard shell", "four wheels", "name tag", "combination lock", "belt strap"]},
}
CATEGORIES = list(CATEGORY_SPECS)

PLACE_HINTS = ["on the train", "near the escalator", "on the platform", "near gate 2", "in the last coach",
               "in the women's coach", "near the security check", "in the lift", "near the ticket counter",
               "on a seat", "near the stairs"]
LOST_TEMPLATES = ["Lost my {core} with {feat}", "I lost a {core}, it has {feat}",
                  "Missing {core} containing {feat}", "{core} lost, {feat}",
                  "Please help, lost {core} having {feat}", "Left my {core} in the metro, {feat}"]
FOUND_TEMPLATES = ["Found a {core} with {feat}", "{core} found, it has {feat}",
                   "Found {core} containing {feat}", "Picked up a {core}, {feat}",
                   "Someone left a {core} having {feat}", "{core} found on seat, {feat}"]

POS_HOP_WEIGHTS = {0: .30, 1: .20, 2: .14, 3: .10, 4: .07, 5: .05, 6: .04, 7: .03, 8: .03}
BUSY = ["Rajiv Chowk", "Kashmere Gate", "New Delhi", "Hauz Khas", "Central Secretariat", "HUDA City Centre",
        "Noida City Centre", "Anand Vihar", "Dwarka Sector 21", "Lajpat Nagar", "Hauz Khas", "Netaji Subhash Place"]


def color_word(rng, color):
    return rng.choice(COLOR_WORDS.get(color, [color]))


def random_item(rng, category=None):
    category = category or rng.choice(CATEGORIES)
    spec = CATEGORY_SPECS[category]
    brand = rng.choice(spec["brands"]) if spec["brands"] and rng.random() < 0.7 else ""
    return {"category": category, "color": rng.choice(COLORS), "brand": brand,
            "features": rng.sample(spec["features"], k=rng.choice([1, 2]))}


def add_noise(rng, text):
    """Typos, odd capitalisation, special characters and extra spaces."""
    words = text.split()
    if rng.random() < 0.15:
        i = rng.randrange(len(words))
        w = words[i]
        if len(w) > 3:
            j = rng.randrange(len(w) - 1)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    text = " ".join(words)
    r = rng.random()
    if r < 0.15:
        text = text.upper()
    elif r < 0.30:
        text = text.lower()
    if rng.random() < 0.2:
        text += rng.choice(["!!", "...", "!!!", "@@", " #urgent"])
    if rng.random() < 0.1:
        text = text.replace(" ", "  ", 1)
    return text


def render_description(rng, item, role):
    """Turn a hidden item into free text, the way a loser or a finder might write it."""
    spec = CATEGORY_SPECS[item["category"]]
    noun = rng.choice(spec["nouns"])
    brand = item["brand"] if item["brand"] and rng.random() < 0.65 else ""
    core = " ".join(x for x in [color_word(rng, item["color"]), brand, noun] if x)
    feats = [f for f in item["features"] if rng.random() < 0.7]
    if not feats and rng.random() < 0.6:
        feats = [rng.choice(item["features"])]
    if feats:
        text = rng.choice(LOST_TEMPLATES if role == "lost" else FOUND_TEMPLATES).format(
            core=core, feat=" and ".join(feats))
    else:
        text = rng.choice(["Lost my {core}", "Missing {core}"] if role == "lost"
                          else ["Found a {core}", "{core} found"]).format(core=core)
    if rng.random() < 0.3:
        text += " " + rng.choice(PLACE_HINTS)
    return add_noise(rng, text)


def make_hard_negative(rng, item):
    """Same category but a different colour or brand (and sometimes different details)."""
    spec = CATEGORY_SPECS[item["category"]]
    other = {**item, "features": list(item["features"])}
    if rng.random() < 0.7 or not spec["brands"]:
        other["color"] = rng.choice([c for c in COLORS if c != item["color"]])
    else:
        other["brand"] = rng.choice([b for b in spec["brands"] if b != item["brand"]])
    if rng.random() < 0.5:
        other["features"] = rng.sample(spec["features"], k=rng.choice([1, 2]))
    return other


# ---- stations and dates ----------------------------------------------------------------------------------
def sample_near_station(rng, station):
    """Where a found item was handed in, given where it was lost (matching pairs)."""
    if rng.random() < 0.04:  # item rode a long way before somebody found it
        far = [s for s, h in st.HOPS[station].items() if h >= 9]
        return rng.choice(far) if far else station
    d = rng.choices(list(POS_HOP_WEIGHTS), list(POS_HOP_WEIGHTS.values()))[0]
    cands = [s for s, h in st.HOPS[station].items() if h == d]
    return rng.choice(cands) if cands else station


def sample_station_pair_negative(rng, station):
    if rng.random() < 0.25:
        near = [s for s, h in st.HOPS[station].items() if h <= 3]
        return rng.choice(near)
    return rng.choice(st.STATIONS)


def sample_gap_days_positive(rng):
    r = rng.random()
    if r < .50:
        return 0
    if r < .70:
        return 1
    if r < .80:
        return 2
    if r < .95:
        return rng.randint(3, 7)
    return rng.randint(8, 14)


def lost_category_noise(rng, true_cat):
    r = rng.random()
    if r < .88:
        return true_cat
    if r < .95:
        return UNKNOWN
    return rng.choice([c for c in CATEGORIES if c != true_cat])


def found_category_noise(rng, true_cat):
    return true_cat if rng.random() < .92 else UNKNOWN


# ---- datasets --------------------------------------------------------------------------------------------
def make_training_data(rng, n=2000, base="2026-09-01"):
    base = pd.Timestamp(base)
    rows = []
    for i in range(n):
        is_match = i % 2 == 0
        item = random_item(rng)
        lost_station = rng.choice(st.STATIONS)
        lost_date = base + pd.Timedelta(days=rng.randint(0, 40))
        if is_match:
            other = item
            found_station = sample_near_station(rng, lost_station)
            gap = sample_gap_days_positive(rng)
        else:
            if rng.random() < 0.5:
                other = make_hard_negative(rng, item)
            else:
                other = random_item(rng, rng.choice([c for c in CATEGORIES if c != item["category"]]))
            found_station = sample_station_pair_negative(rng, lost_station)
            gap = rng.randint(-10, 14)
        rows.append({
            "lost_description": render_description(rng, item, "lost"),
            "found_description": render_description(rng, other, "found"),
            "lost_category": lost_category_noise(rng, item["category"]),
            "found_category": found_category_noise(rng, other["category"]),
            "lost_station": lost_station, "found_station": found_station,
            "lost_date": lost_date.date().isoformat(),
            "found_date": (lost_date + pd.Timedelta(days=gap)).date().isoformat(),
            "match": int(is_match)})
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


def make_found_database(rng, n=500):
    weights = [6 if s in BUSY else (3 if st.is_interchange(s) else 1) for s in st.STATIONS]
    day_w = [10, 9, 8, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 2, 1]
    items, rows = [], []
    for i in range(n):
        item = random_item(rng)
        items.append(item)
        rows.append({"item_id": f"F{i + 1:03d}",
                     "description": render_description(rng, item, "found"),
                     "category": found_category_noise(rng, item["category"]),
                     "station": rng.choices(st.STATIONS, weights)[0],
                     "days_ago": rng.choices(range(15), day_w)[0]})
    return pd.DataFrame(rows), items


def make_query_for(rng, item, found_row, base_date):
    """A fresh lost report written about a known found item, used to test the search end to end."""
    gap = sample_gap_days_positive(rng)
    found_date = base_date - pd.Timedelta(days=int(found_row["days_ago"]))
    return {"description": render_description(rng, item, "lost"),
            "category": lost_category_noise(rng, item["category"]),
            "station": sample_near_station(rng, found_row["station"]),
            "date": (found_date - pd.Timedelta(days=gap)).date().isoformat()}


# =====================================================================================================
# STRICT MODE
# The first version of this data had shortcuts a classifier could lean on: every matching pair was found on or
# after the loss date, matches were always close in place, and most non-matches were obviously different items.
# Strict mode removes those shortcuts and adds the kinds of confusion real reports have.
# =====================================================================================================
STRICT = {
    "hard_share": 0.45,         # non-matches that are look-alikes: same category, often same colour
    "hard_same_color": 0.28,    # among look-alikes, share that keep the same colour (differ in brand or contents)
    "twin_share": 0.03,         # non-matches that are the SAME kind of item (two black phones), text cannot tell
    "color_confusion": 0.06,    # matches where the two reporters name different colours (black vs navy)
    "vague_share": 0.07,        # reports that give almost no detail ("lost my bag")
    "drop_feat": 0.25,          # chance a reporter leaves out each detail
    "neg_near_station": 0.55,   # non-matches whose stations are as close as matches (no location shortcut)
    "lost_cat_unknown": 0.08, "lost_cat_wrong": 0.04,
    "found_cat_unknown": 0.07, "found_cat_wrong": 0.03,
}
CONFUSABLE = {"black": ["navy", "grey", "brown"], "blue": ["navy", "green", "grey"],
              "red": ["maroon", "pink", "brown"], "green": ["blue", "grey"],
              "brown": ["maroon", "black", "yellow"], "white": ["silver", "grey", "yellow"],
              "grey": ["silver", "black", "white"], "silver": ["grey", "white"], "pink": ["red", "white"],
              "yellow": ["white", "brown"], "navy": ["blue", "black"], "maroon": ["red", "brown"]}


def signature(item):
    """One string that identifies an item's exact attributes; used to keep train and test items apart."""
    return "|".join([item["category"], item["color"], item["brand"], ",".join(sorted(item["features"]))])


def render_strict(rng, item, role, cfg=STRICT, confuse=False):
    item = dict(item)
    if confuse:
        item["color"] = rng.choice(CONFUSABLE[item["color"]])
    spec = CATEGORY_SPECS[item["category"]]
    noun = rng.choice(spec["nouns"])
    if rng.random() < cfg["vague_share"]:
        core = rng.choice([noun, f"{color_word(rng, item['color'])} {noun}"])
        text = rng.choice(["Lost my {c}", "Missing {c}"] if role == "lost"
                          else ["Found a {c}", "{c} found"]).format(c=core)
    else:
        brand = item["brand"] if item["brand"] and rng.random() < 0.55 else ""
        core = " ".join(x for x in [color_word(rng, item["color"]), brand, noun] if x)
        feats = [f for f in item["features"] if rng.random() > cfg["drop_feat"]]
        if feats:
            text = rng.choice(LOST_TEMPLATES if role == "lost" else FOUND_TEMPLATES).format(
                core=core, feat=" and ".join(feats))
        else:
            text = rng.choice(["Lost my {c}", "Missing {c}"] if role == "lost"
                              else ["Found a {c}", "{c} found"]).format(c=core)
    if rng.random() < 0.3:
        text += " " + rng.choice(PLACE_HINTS)
    return add_noise(rng, text)


def hard_negative_strict(rng, item, cfg=STRICT):
    """A different item of the same category, often the same colour, differing in brand or contents."""
    spec = CATEGORY_SPECS[item["category"]]
    other = {**item, "features": list(item["features"])}
    keep_color = rng.random() < cfg["hard_same_color"]
    if not keep_color:
        other["color"] = rng.choice([c for c in COLORS if c != item["color"]])
    if spec["brands"] and (keep_color or rng.random() < 0.3):
        other["brand"] = rng.choice([b for b in [""] + spec["brands"] if b != item["brand"]])
    if keep_color or rng.random() < 0.5:
        for _ in range(8):
            other["features"] = rng.sample(spec["features"], k=rng.choice([1, 2]))
            if sorted(other["features"]) != sorted(item["features"]):
                break
    return other


def _cat_noise(rng, true_cat, p_unknown, p_wrong):
    r = rng.random()
    if r < p_unknown:
        return UNKNOWN
    if r < p_unknown + p_wrong:
        return rng.choice([c for c in CATEGORIES if c != true_cat])
    return true_cat


def make_training_data_strict(rng, n=2000, base="2026-09-01", cfg=STRICT):
    base = pd.Timestamp(base)
    rows = []
    for i in range(n):
        is_match = i % 2 == 0
        item = random_item(rng)
        lost_station = rng.choice(st.STATIONS)
        lost_date = base + pd.Timedelta(days=rng.randint(0, 40))
        gap = sample_gap_days_positive(rng)  # the SAME date pattern for matches and non-matches
        if is_match:
            other, found_station = item, sample_near_station(rng, lost_station)
            confuse = rng.random() < cfg["color_confusion"]
            lost_desc = render_strict(rng, item, "lost")
            found_desc = render_strict(rng, other, "found", confuse=confuse)
        else:
            r = rng.random()
            if r < cfg["twin_share"]:
                other = item
            elif r < cfg["twin_share"] + cfg["hard_share"]:
                other = hard_negative_strict(rng, item, cfg)
            elif rng.random() < 0.5:
                other = random_item(rng, rng.choice([c for c in CATEGORIES if c != item["category"]]))
            else:
                other = random_item(rng, item["category"])
            found_station = (sample_near_station(rng, lost_station) if rng.random() < cfg["neg_near_station"]
                             else rng.choice(st.STATIONS))
            lost_desc, found_desc = render_strict(rng, item, "lost"), render_strict(rng, other, "found")
        rows.append({
            "pair_id": i, "item_group": signature(item),
            "lost_description": lost_desc, "found_description": found_desc,
            "lost_category": _cat_noise(rng, item["category"], cfg["lost_cat_unknown"], cfg["lost_cat_wrong"]),
            "found_category": _cat_noise(rng, other["category"], cfg["found_cat_unknown"], cfg["found_cat_wrong"]),
            "lost_station": lost_station, "found_station": found_station,
            "lost_date": lost_date.date().isoformat(),
            "found_date": (lost_date + pd.Timedelta(days=gap)).date().isoformat(),
            "match": int(is_match)})
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


def make_found_database_strict(rng, n=500, exclude_groups=frozenset(), cfg=STRICT):
    """Found items whose exact attribute combination never appears in the training data."""
    weights = [6 if s in BUSY else (3 if st.is_interchange(s) else 1) for s in st.STATIONS]
    day_w = [10, 9, 8, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 2, 1]
    items, rows = [], []
    for i in range(n):
        for _ in range(100):
            item = random_item(rng)
            if signature(item) not in exclude_groups:
                break
        items.append(item)
        rows.append({"item_id": f"F{i + 1:03d}", "description": render_strict(rng, item, "found"),
                     "category": _cat_noise(rng, item["category"], cfg["found_cat_unknown"], cfg["found_cat_wrong"]),
                     "station": rng.choices(st.STATIONS, weights)[0],
                     "days_ago": rng.choices(range(15), day_w)[0]})
    return pd.DataFrame(rows), items


def make_query_strict(rng, item, found_row, base_date, cfg=STRICT):
    gap = sample_gap_days_positive(rng)
    found_date = base_date - pd.Timedelta(days=int(found_row["days_ago"]))
    return {"description": render_strict(rng, item, "lost", confuse=rng.random() < cfg["color_confusion"]),
            "category": _cat_noise(rng, item["category"], cfg["lost_cat_unknown"], cfg["lost_cat_wrong"]),
            "station": sample_near_station(rng, found_row["station"]),
            "date": (found_date - pd.Timedelta(days=gap)).date().isoformat()}
