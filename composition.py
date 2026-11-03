"""Turn a chemical formula (e.g. La0.6Sr0.4Co0.2Fe0.8O3) into the model's descriptors.

Element values are not typed in: train.py recovers them from the dataset, because every
site-level descriptor there is an exact weighted average of per-element values
(A-site and B-site fractions). Only elements that occur in the training data are supported.
"""
import json
import math
import re

import numpy as np

PROPS_A = ["Ion", "EN", "R", "Sa", "SmithAcid", "Ionic"]
PROPS_B = PROPS_A + ["Delec"]
R_OXYGEN = 1.40  # reproduces the dataset's tolerance factor t exactly
COMP_FEATURES = [
    "Ion_A", "Ion_B", "EN_A", "EN_B", "R_A", "R_B", "Sa_A", "Sa_B", "Delec_B", "t",
    "Ionic_A", "Ionic_B", "SmithAcid_A", "SmithAcid_B", "1000/T",
]
_NUM = r"\d*\.\d+|\d+"


def parse_formula(text):
    s = re.sub(r"\s+", "", str(text))
    if not s:
        raise ValueError("Type a material formula, for example La0.6Sr0.4Co0.2Fe0.8O3.")
    toks = re.findall(r"\(|\)|[A-Z][a-z]?|" + _NUM, s)
    if "".join(toks) != s:
        raise ValueError("Could not read the formula. Use element symbols and numbers only, like LaCoO3.")
    pos = 0

    def number():
        nonlocal pos
        if pos < len(toks) and re.fullmatch(_NUM, toks[pos]):
            pos += 1
            return float(toks[pos - 1])
        return 1.0

    def group():
        nonlocal pos
        out = []
        while pos < len(toks) and toks[pos] != ")":
            t = toks[pos]
            if t == "(":
                pos += 1
                inner = group()
                if pos >= len(toks):
                    raise ValueError("The formula has an opening bracket without a closing one.")
                pos += 1
                m = number()
                out += [(e, a * m) for e, a in inner]
            elif re.fullmatch(r"[A-Z][a-z]?", t):
                pos += 1
                out.append((t, number()))
            else:
                raise ValueError("Could not read the formula. A number must follow an element symbol.")
        return out

    items = group()
    if pos != len(toks):
        raise ValueError("The formula has a closing bracket without an opening one.")
    merged = {}
    for e, a in items:
        if e != "O":
            merged[e] = merged.get(e, 0.0) + a
    return list(merged.items())


def parse_site(text):
    out = {}
    for part in str(text).split(","):
        m = re.fullmatch(r"\s*([A-Z][a-z]?)(\d*\.?\d*)\s*", part)
        out[m.group(1)] = float(m.group(2) or 1)
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}


def fit_tables(df):
    """Recover per-element values from the training table (needs columns A and B)."""
    sites = {"A": [parse_site(s) for s in df["A"]], "B": [parse_site(s) for s in df["B"]]}
    tables = {}
    for site, props in (("A", PROPS_A), ("B", PROPS_B)):
        els = sorted({e for row in sites[site] for e in row})
        M = np.array([[row.get(e, 0.0) for e in els] for row in sites[site]])
        tables[site] = {e: {} for e in els}
        for p in props:
            w = np.linalg.lstsq(M, df[f"{p}_{site}"].values, rcond=None)[0]
            for e, v in zip(els, w):
                tables[site][e][p] = float(v)
        _, s, vt = np.linalg.svd(M)
        rank = int((s > 1e-9).sum())
        null = vt[rank:]
        tables["amb_" + site] = [e for i, e in enumerate(els) if null.size and np.abs(null[:, i]).max() > 1e-6]
    if "Material composition" in df.columns:
        tables["known"] = {
            re.sub(r"\s+", "", str(m)): {"A": str(a), "B": str(b)}
            for m, a, b in zip(df["Material composition"], df["A"], df["B"])
        }
    return tables


def split_sites(items, tables):
    a_set, b_set = set(tables["A"]), set(tables["B"])
    A, B, seen_b = {}, {}, False
    for e, amt in items:
        if e in a_set and e in b_set:
            site = "B" if seen_b else "A"
        elif e in a_set:
            site = "A"
        elif e in b_set:
            site = "B"
        else:
            raise ValueError(
                f"{e} is not in the training data. A-site elements: {', '.join(sorted(a_set - b_set | a_set & b_set))}. "
                f"B-site elements: {', '.join(sorted(b_set))}."
            )
        if site == "B":
            seen_b = True
        (A if site == "A" else B)[e] = amt
    if not A or not B:
        raise ValueError("Give both A-site and B-site elements, with the A-site first, like La0.6Sr0.4CoO3.")
    return ({k: v / sum(A.values()) for k, v in A.items()}, {k: v / sum(B.values()) for k, v in B.items()})


def descriptors(formula, temp_c, tables):
    key = re.sub(r"\s+", "", str(formula))
    known = tables.get("known", {}).get(key)
    if known:  # a dataset material: use the dataset's own A/B split
        A, B = parse_site(known["A"]), parse_site(known["B"])
    else:
        items = parse_formula(formula)
        m = re.search(r"O(\d*\.?\d*)$", key)
        o = float(m.group(1) or 1) if m else None
        if o is not None and abs(o - 3) > 1e-6:
            raise ValueError(f"Only ABO3 perovskites can be entered by name; this formula has O{o:g}. Use the descriptor form instead.")
        A, B = split_sites(items, tables)
    feats = {}
    for site, fr, props in (("A", A, PROPS_A), ("B", B, PROPS_B)):
        for p in props:
            feats[f"{p}_{site}"] = sum(f * tables[site][e][p] for e, f in fr.items())
    feats["t"] = (feats["R_A"] + R_OXYGEN) / (math.sqrt(2) * (feats["R_B"] + R_OXYGEN))
    feats["1000/T"] = 1000.0 / (float(temp_c) + 273.15)
    warns = [
        f"{e} on the {s}-site is uncertain: the training data never varied it independently."
        for s, fr in (("A", A), ("B", B)) for e in fr if e in tables["amb_" + s]
    ]
    return {k: feats[k] for k in COMP_FEATURES}, {"A": A, "B": B}, warns
