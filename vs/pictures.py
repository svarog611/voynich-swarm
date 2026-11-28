"""The closed picture vocabulary (P-L4-030, P-L4-032) and how two annotations of a folio are compared.

Free wording cannot agree as strings: two seats who both see a figure standing in a tub score zero
on 'posture' unless they wrote the same sentence. So agreement is measured only on a fixed list of
features with a fixed set of values. Any other feature may still be written; it is kept on the
record as description and never compared.

New annotations must write vocabulary features with one of the listed values (vs/experiment.py
checks this). Annotations already on the record were written before the list existed; their values
are read through `normalize`, which maps the common wordings onto the list and leaves the rest out
of the comparison (a value it cannot place is not compared, never counted as a disagreement).

For visual agreement, objects are paired by label locus where both annotations give one; otherwise within the same kind
(plant, plant_part, root, leaf and flower count as one kind, and so do jar and container) and ring, by nearest clock angle when both give one, else in list order.
Label comparisons have a separate, label-blind pairing rule (compare_labels).
"""
import re
from collections import Counter
from dataclasses import dataclass
from math import isclose, isfinite

UNCLEAR = "unclear"

# feature -> allowed values. Every feature also accepts "unclear", which is never compared.
VOCAB = {
    "sex": ("female", "male"),
    "clothing": ("nude", "partly", "clothed"),
    "container": ("none", "tub", "barrel", "other"),
    "posture": ("standing", "lying", "in_container", "sitting"),
    "headwear": ("none", "some"),
    "holds": ("nothing", "star", "other"),
    "hair": ("long", "short", "bound"),
    "cheeks": ("red", "none"),
    "facing": ("left", "right", "front"),
    "ring": ("outer", "middle", "inner", "centre", "outside"),
    "colour": ("none", "red", "blue", "green", "yellow", "brown", "mixed"),
    "points": tuple(str(n) for n in range(3, 33)),
    "centre": ("none", "dot", "ring", "face"),
    "tail": ("yes", "no"),
    "count": tuple(str(n) for n in range(1, 101)),
}

# older or alternative key names on the record -> vocabulary key
ALIASES = {"color": "colour", "headdress": "headwear", "headgear": "headwear", "nude": "clothing",
           "star_points": "points", "rays": "points", "centre_mark": "centre", "center_mark": "centre",
           "star": "holds"}

_COLOURS = (("red", ("red", "pink", "rose")), ("blue", ("blue",)), ("green", ("green", "olive")),
            ("yellow", ("yellow", "ochre", "tan", "gold")), ("brown", ("brown",)))


def _words(v):
    return re.findall(r"[a-z]+", v)


def _colour(v):
    if v.startswith(("uncoloured", "uncolored", "ink only", "none", "outline", "unfilled", "no colour")):
        return "none"
    found = {c for c, ws in _COLOURS if any(w.startswith(x) for w in _words(v) for x in ws)}
    if len(found) == 1:
        return found.pop()
    return "mixed" if found else None


def _points(v):
    ns = sorted({int(n) for n in re.findall(r"\d+", v)})
    return str(ns[0]) if len(ns) == 1 else None   # "7 or 8" is not one value: not compared


def _normalize(key, v):
    w = set(_words(v))
    if key == "sex":
        return "female" if "female" in w or "woman" in w else "male" if "male" in w or "man" in w else None
    if key == "clothing":
        if {"dressed", "robe", "tunic", "clothed", "garment", "gown"} & w:
            return "clothed"
        if "bare" in w or "above" in w or "partly" in w:
            return "partly"
        return "nude" if "nude" in w or "naked" in w else None
    if key == "container":
        if v.startswith(("none", "no ")) or v == "no":
            return "none"
        return "barrel" if "barrel" in w else "tub" if "tub" in w else None
    if key == "posture":
        if {"tub", "barrel", "emerging", "rising", "half"} & w or "upper body" in v:
            return "in_container"
        if "lying" in w or "reclining" in w:
            return "lying"
        if "sitting" in w or "seated" in w:
            return "sitting"
        return "standing" if "standing" in w or "upright" in w else None
    if key == "headwear":
        if v in ("no", "none", "bare", "false"):
            return "none"
        return "some" if {"yes", "band", "crown", "cap", "hat", "wreath", "veil", "true"} & w else None
    if key == "holds":
        if v in ("yes", "true", "attached") or v.startswith("star"):
            return "star"
        if v in ("no", "none", "false", "nothing"):
            return "nothing"
        return None
    if key == "hair":
        if {"bound", "braided", "braids", "up"} & w:
            return "bound"
        return "long" if "long" in w else "short" if "short" in w else None
    if key == "cheeks":
        return "red" if "red" in w else "none" if v in ("none", "no", "plain") else None
    if key == "ring":
        if {"top", "outside", "above"} & w:
            return "outside"
        for r in ("outer", "middle", "inner"):
            if r in w:
                return r
        return "centre" if {"centre", "center"} & w else None
    if key == "colour":
        return _colour(v)
    if key in ("points", "count"):
        p = _points(v)
        return p if p in VOCAB[key] else None
    if key == "centre":
        if v.startswith("none") or v in ("no", "hollow", "empty"):
            return "none"
        if "dot" in w:
            return "dot"
        if "ring" in w or "circle" in w or "ringed" in w:
            return "ring"
        return "face" if "face" in w else None
    if key == "tail":
        return "yes" if v in ("yes", "true") else "no" if v in ("no", "none", "false") else None
    if key == "facing":
        for d in ("left", "right"):
            if d in w:
                return d
        return "front" if {"front", "frontal", "viewer"} & w else None
    return None


def normalize(features):
    """The comparable vocabulary features of one object: {key: value}. Exact vocabulary values are
    taken as written; older wordings are mapped where the mapping is plain; the rest is dropped."""
    out = {}
    for k, v in (features or {}).items():
        key = ALIASES.get(k, k)
        if key not in VOCAB or key in out and k != key:
            continue
        s = str(v).strip().lower()
        if s in VOCAB[key]:
            val = s
        elif s == UNCLEAR:
            continue
        else:
            val = _normalize(key, s)
        if val is not None:
            out[key] = val
    return out


def check_features(features):
    """Errors for a new annotation: a vocabulary feature must carry one of its listed values."""
    errs = []
    for k, v in (features or {}).items():
        key = ALIASES.get(k, k)
        if key not in VOCAB:
            continue
        if k != key:
            errs.append(f"feature {k!r}: write it as {key!r}")
        s = str(v).strip().lower()
        if s != UNCLEAR and s not in VOCAB[key]:
            vals = VOCAB[key] if len(VOCAB[key]) < 12 else (VOCAB[key][0] + ".." + VOCAB[key][-1],)
            errs.append(f"feature {key!r} is one of {', '.join(vals)} or unclear (got {v!r}); free description "
                        "goes under any other feature name, where it is kept but not compared")
    return errs


def _angle(o):
    f = o.get("features") or {}
    for k in ("angle_deg_clockwise_from_top", "bearing_clockwise_deg"):
        try:
            return float(f[k]) % 360
        except (KeyError, TypeError, ValueError):
            pass
    for k in ("clock_position", "clock"):
        m = re.match(r"^\s*(\d{1,2})(?::(\d{2}))?", str(f.get(k, "")))
        if m:
            h, mi = int(m.group(1)) % 12, int(m.group(2) or 0)
            return (h * 30 + mi / 2) % 360
    return None


def _ring(o):
    return normalize({"ring": (o.get("features") or {}).get("ring", "")}).get("ring")


def clock_angle(obj):
    """An explicitly reported bearing, including legacy clock readings. Ordinals are not bearings."""
    fs = obj.get("features") or {}
    for key in ("angle_deg_clockwise_from_top", "bearing_clockwise_deg"):
        value = fs.get(key)
        if value is not None and not isinstance(value, bool):
            try:
                degrees = float(value)
                if isfinite(degrees):
                    return degrees % 360
            except (TypeError, ValueError):
                pass
    position = re.search(r"\(about (\d+(?:\.\d+)?) o'clock\)$", str(fs.get("position", "")), re.I)
    for clock in (fs.get("clock_position"), fs.get("clock"), position[1] if position else None):
        m = re.fullmatch(r"(\d{1,2}(?:\.\d+)?)(?::([0-5]\d))?", str(clock).strip())
        if m and float(m[1]) <= 12:
            return (float(m[1]) * 30 + int(m[2] or 0) / 2) % 360
    return None


def circular_distance(a, b):
    return abs((a - b + 180) % 360 - 180)


def unique_nearest(candidates, tolerance):
    """Return the sole nearest candidate within tolerance, leaving ties unresolved."""
    ranked = sorted(candidates, key=lambda item: item[0])
    if not ranked or ranked[0][0] > tolerance:
        return None
    if len(ranked) > 1 and isclose(ranked[0][0], ranked[1][0], abs_tol=1e-9):
        return None
    return ranked[0][1]


@dataclass
class _Location:
    kind: str
    ring: str | None
    row: str | None
    angle: float | None
    features: dict

    def requires_bearing(self):
        return self.ring in ("outer", "middle", "inner") or (self.ring != "centre" and self.angle is not None)

    def compatible(self, other):
        return self.kind == other.kind and not self.different_region(other)

    def different_region(self, other):
        return bool((self.ring and other.ring and self.ring != other.ring)
                    or (self.row and other.row and self.row != other.row))

    def distinct_from(self, other):
        return self.different_region(other) or (
            self.requires_bearing() and other.requires_bearing()
            and self.angle is not None and other.angle is not None
            and circular_distance(self.angle, other.angle) > MAX_OBJECT_DISTANCE_DEG
        )


def _locations(detail):
    rows_described = re.search(r"\brows?\b", str(detail.get("statement", "")), re.I)
    out = {}
    for i, o in enumerate(detail.get("objects") or []):
        if not isinstance(o, dict) or not o.get("kind"):
            continue
        # Older non-circular catalogues identify the row in IDs such as row1-jar and r1-jar.
        row = re.match(r"^(?:row|r)([1-9]\d*)[-_]", str(o.get("id", "")), re.I) if rows_described else None
        features = normalize(o.get("features"))
        # the literal kind: for a label link, a root and a plant are not shown to be the same drawing
        out[i] = _Location(o["kind"], features.pop("ring", None), row[1] if row else None,
                           clock_angle(o), features)
    return out


MAX_OBJECT_DISTANCE_DEG = 30


def _partner(location, candidates):
    if location.requires_bearing() or any(o.requires_bearing() for _, o in candidates):
        # A missing bearing on a rival cannot establish which drawing is nearest.
        if not location.ring or location.angle is None or any(o.ring != location.ring or o.angle is None for _, o in candidates):
            return None
        return unique_nearest(((circular_distance(location.angle, o.angle), j) for j, o in candidates),
                              MAX_OBJECT_DISTANCE_DEG)
    if len(candidates) == 1:
        return candidates[0][0]
    # For legacy drawings without clock positions, only a recorded difference rules out a rival.
    possible = [(j, o) for j, o in candidates if all(o.features.get(k, v) == v for k, v in location.features.items())]
    if len(possible) == 1:
        j, other = possible[0]
        if location.features.keys() & other.features.keys():
            return j
    return None


def _partners(left, right):
    return {i: _partner(o, [(j, q) for j, q in right.items() if o.compatible(q)]) for i, o in left.items()}


def explicit_label(obj):
    value = obj.get("label") if isinstance(obj, dict) else None
    return (value.strip() or None) if isinstance(value, str) else None


@dataclass(frozen=True)
class LabelPair:
    left: int
    right: int
    explicit: str
    locus: str | None


def compare_labels(a, b, projected_a=None, projected_b=None):
    """Compare labels only after both objects independently choose one another by location/kind.

    A missing explicit vote counts against explicit agreement. Two omissions abstain.
    A geometric confirmation requires both positions to identify the same line.
    """
    la, lb = _locations(a), _locations(b)
    ab, ba = _partners(la, lb), _partners(lb, la)
    labels_a, labels_b = ([explicit_label(o) for o in d["objects"]] for d in (a, b))
    counts_a, counts_b = Counter(labels_a), Counter(labels_b)
    pairs = []
    for i, j in ab.items():
        if j is None or ba.get(j) != i:
            continue
        left_label, right_label = labels_a[i], labels_b[j]
        locus = None
        if left_label and right_label:
            explicit = "same" if left_label == right_label else "different"
            if left_label == right_label:
                locus = left_label
        else:
            explicit_locus = left_label or right_label
            explicit = "one_missing" if explicit_locus else "both_missing"
            projected_locus = (projected_a or {}).get(i)
            if projected_locus and projected_locus == (projected_b or {}).get(j) and explicit_locus in (None, projected_locus):
                locus = projected_locus
        if locus and (counts_a[locus] > int(left_label == locus) or counts_b[locus] > int(right_label == locus)):
            locus = None
        pairs.append(LabelPair(i, j, explicit, locus))
    return pairs


def explicit_label_agreement(pairs):
    votes = [p for p in pairs if p.explicit != "both_missing"]
    return sum(p.explicit == "same" and p.locus is not None for p in votes) / len(votes) if votes else None


def conflicting_unpaired_loci(a, b, pairs):
    """Loci assigned by b to unmatched drawings at a known different position from a's object."""
    matched = {pair.right for pair in pairs}
    claims = [(label, location) for j, location in _locations(b).items()
              if j not in matched and (label := explicit_label(b["objects"][j]))]
    return {(i, label) for i, location in _locations(a).items() for label, other in claims
            if location.distinct_from(other)}


# Seats name one drawing by whichever part they take it for: the same pharmaceutical-page drawing is
# a plant to one seat, a root or a plant_part to the next, and a jar is a container. Pairing and
# agreement treat these as one kind of thing, so a difference of name alone does not leave two
# drawings of the same object unpaired (each unpaired object counts as a disagreement). What the
# seats see is still compared, feature by feature.
KIND_CLASS = {**dict.fromkeys(("plant", "plant_part", "root", "leaf", "flower"), "plant"),
              **dict.fromkeys(("jar", "container"), "jar")}


def kind_class(kind):
    return KIND_CLASS.get(kind, kind)


def measured_angle(obj):
    """An angle in degrees that the seat measured on the scan. A clock hour, or an angle worked out
    from the figure's place in the count, is not a measurement and returns None."""
    fs = obj.get("features") or {}
    for key in ("angle_deg_clockwise_from_top", "bearing_clockwise_deg"):
        value = fs.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value):
            return float(value) % 360
    return None


def pair_objects(A, B):
    """Pairs of objects that describe the same drawing, in A's order."""
    pairs, used = [], set()
    for o in A:
        lab = o.get("label")
        m = next((k for k, q in enumerate(B) if k not in used and lab and q.get("label") == lab), None)
        if m is None:
            cands = [k for k, q in enumerate(B) if k not in used and kind_class(q.get("kind")) == kind_class(o.get("kind"))
                     and not (lab and q.get("label"))
                     and (_ring(o) is None or _ring(q) is None or _ring(o) == _ring(q))]
            ao = _angle(o)
            angled = [(min(abs(ao - aq), 360 - abs(ao - aq)), k) for k in cands
                      for aq in [_angle(B[k])] if ao is not None and aq is not None]
            if angled:
                d, m = min(angled)
                m = m if d <= 30 else None   # more than an hour apart on the clock is not the same figure
            elif cands:
                m = cands[0]
        if m is not None:
            used.add(m)
            pairs.append((o, B[m]))
    return pairs


def agreement(a, b):
    """Share of compared items on which two annotations of one folio agree. Compared items: every
    vocabulary feature both recorded on a paired object; for objects paired by label locus, also
    their kind and ring (objects paired without a label were paired on kind and ring, so those
    cannot count as agreement); and every object one seat drew that the other has no partner for,
    which counts as a disagreement. None if nothing is comparable."""
    def objs(d):
        return [o for o in (d.get("objects") or []) if isinstance(o, dict)]
    A, B = objs(a), objs(b)
    pairs = pair_objects(A, B)
    tot = len(A) + len(B) - 2 * len(pairs)
    same = 0
    for o, q in pairs:
        fo, fq = normalize(o.get("features")), normalize(q.get("features"))
        by_label = bool(o.get("label")) and o.get("label") == q.get("label")
        if by_label:
            tot += 1
            same += kind_class(o.get("kind")) == kind_class(q.get("kind"))
        else:
            fo.pop("ring", None)
            fq.pop("ring", None)
        for k in set(fo) & set(fq):
            tot += 1
            same += fo[k] == fq[k]
    return same / tot if tot else None
