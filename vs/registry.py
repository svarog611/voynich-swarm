"""The record: every fact, grid cell, anchor, audit, attack, segmentation and step proposal,
each added by an agent through a pull request and judged by the scorer.

registry/items.jsonl holds one item per line (schema in the docstring of add()). Nothing is
seeded: the record starts empty and only the scorer writes to it, from a pull request's
experiment files. Items are never edited after they are judged; a sharper claim is a new item.
"""
import datetime, json, os
from . import path

FN = path("registry", "items.jsonl")


WITHDRAWN = path("registry", "withdrawn.jsonl")
PICTURE_AGREED = 0.8


def picture_seat(record):
    provenance = record.get("provenance") or {}
    return provenance.get("seat", provenance.get("added_by"))


def read(type=None, include_withdrawn=False):
    """Items on the record. Withdrawals (pipeline errors only, each with a public reason in
    registry/withdrawn.jsonl) are applied as an overlay; the original lines are never edited."""
    if not os.path.exists(FN):
        return []
    items = [json.loads(l) for l in open(FN) if l.strip()]
    wd = {}
    if os.path.exists(WITHDRAWN):
        for l in open(WITHDRAWN):
            if l.strip():
                w = json.loads(l)
                wd[w["id"]] = w
    from .identity import owner, agent
    out = []
    for i in items:
        pv = i.get("provenance", {})
        if pv.get("added_by") and "seat" not in pv:
            seat = pv["added_by"]
            i["provenance"] = dict(pv, seat=seat, added_by=owner(seat), agent=agent(seat, pv.get("agent")))
            if owner(seat) != seat and "@" + seat in i.get("title", ""):
                i["title"] = i["title"].replace("@" + seat, "@" + owner(seat) + " (" + (agent(seat) or seat) + ")")
        if i["id"] in wd:
            if not include_withdrawn:
                continue
            i = dict(i, withdrawn=wd[i["id"]])
        if not i.get("review") or i.get("review") == "not yet reviewed":
            i["review"] = "people review only a full pass"
        if type is None or i["type"] == type:
            out.append(i)
    # anchors: status decided by the swarm (agree / dissent second looks), see judge.anchor_status
    if type in (None, "anchor"):
        from .judge import anchor_status, _look
        anchors = [i for i in out if i["type"] == "anchor"]
        for i in anchors:
            tgt, assess = _look(i)
            if tgt:
                i["verdict"] = {"agree": "AGREE", "dissent": "DISSENT"}.get(assess, "NOTE")
            else:
                i["verdict"] = {"confirmed": "CONFIRMED", "disputed": "DISPUTED"}.get(anchor_status(i, anchors), "DRAFT")
    # pictures: a folio's annotation is AGREED once a second, independent seat annotated the same
    # folio and the two agree on at least 80% of what they can compare on the closed vocabulary
    # (vs/pictures.py: objects matched by label locus, else by kind, ring and clock angle; free
    # wording is never compared; an object only one seat drew counts against); DISAGREE below 50%
    if type in (None, "picture"):
        pics = [i for i in out if i["type"] == "picture"]
        by_folio = {}
        for i in pics:
            by_folio.setdefault(str((i.get("detail") or {}).get("folio")), []).append(i)
        for f, group in by_folio.items():
            for i in group:
                best = None
                for j in group:
                    si, sj = picture_seat(i), picture_seat(j)
                    if j is i or si == sj:
                        continue
                    a_ = picture_agreement(i["detail"], j["detail"])
                    if a_ is not None and (best is None or a_ > best):
                        best = a_
                if best is not None:
                    i["verdict"] = "AGREED" if best >= PICTURE_AGREED else ("DISAGREE" if best < 0.5 else "DRAFT")
                    i["agreement"] = round(best, 3)
    # attacks: a succeeded attack whose generator writes the target pattern in by hand is a weak
    # refutation once someone other than the attacker files a 'weak' second look on it
    if type in (None, "attack"):
        weak = {(i.get("detail") or {}).get("of") for i in (out if type is None else read("source"))
                if i["type"] == "source" and str(i.get("verdict", "")).upper() == "WEAK"}
        for i in out:
            if i["type"] == "attack" and i["verdict"] == "SUCCEEDED" and i["id"] in weak:
                i["verdict"] = "WEAK"
    return out


def fingerprint(item):
    """What makes two submissions the same claim: resubmitting it is refused."""
    d = item.get("detail") or {}
    if item["type"] == "fact":
        if "spec" in d:
            # a spec is the same claim whatever its key order, defaults, thresholds, alpha or permutation
            # count (vs/dsl.py canonical): the hidden value is published with the first verdict, so the
            # same measurement with a moved threshold or a re-rolled null is not a new prediction
            from . import dsl
            try:
                return ("fact", "spec", json.dumps(dsl.canonical(d["spec"]), sort_keys=True))
            except dsl.SpecError:   # a spec scored under older rules: fall back to its exact text
                return ("fact", "spec", json.dumps(d["spec"], sort_keys=True))
        return ("fact", d.get("metric"), json.dumps(d.get("params"), sort_keys=True))
    if item["type"] == "step":
        return ("step", item.get("program"), item.get("title", "").strip().lower())
    if item["type"] == "anchor":
        from .judge import _look
        target, assessment = _look(item)
        if target:
            # Second looks often omit page/voynich. Identify the reviewed anchor,
            # verdict and reviewer, using the same structured/legacy parsing as the judge.
            return ("anchor", "second-look", target, assessment, item["provenance"].get("added_by"))
        return ("anchor", item.get("page"), json.dumps(d.get("voynich")), item["provenance"].get("added_by"))
    return (item["type"], item["provenance"].get("added_by"), json.dumps(item["provenance"].get("files")), item.get("statement"))


def duplicate_of(item):
    if "seat" not in item["provenance"]:
        from .identity import owner
        item = dict(item, provenance=dict(item["provenance"], added_by=owner(item["provenance"]["added_by"])))
    fp = fingerprint(item)
    return next((i["id"] for i in read() if fingerprint(i) == fp), None)


def withdraw(item_id, reason):
    with open(WITHDRAWN, "a") as f:
        f.write(json.dumps({"id": item_id, "reason": reason, "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}) + "\n")


def get(item_id):
    return next((i for i in read(include_withdrawn=True) if i["id"] == item_id), None)


def add(item):
    """item: {id, type, title, statement, verdict, value, detail, program,
    language?, family?, page?, provenance: {added_by, how: "agent", pr, at, files}, review}"""
    os.makedirs(os.path.dirname(FN), exist_ok=True)
    if get(item["id"]):
        raise SystemExit(f"{item['id']} is already on the record; items are never re-judged (submit a new id)")
    dup = duplicate_of(item)
    if dup:
        raise SystemExit(f"DUPLICATE: the same claim is already on the record as {dup}; a claim is scored once")
    item.setdefault("review", "not yet reviewed")
    item["provenance"].setdefault("at", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    item["provenance"].setdefault("how", "agent")
    with open(FN, "a") as f:
        f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    return item


def next_id(prefix):
    import re as _re
    pat = _re.compile(_re.escape(prefix) + r"-(\d+)$")
    nums = [int(m.group(1)) for i in read(include_withdrawn=True) for m in [pat.match(i["id"])] if m]
    return f"{prefix}-{max(nums, default=0) + 1:03d}"


def board_summary():
    """What an agent should know at the start of a session: counts and the latest items."""
    items = read()
    by = {}
    for i in items:
        by.setdefault(i["type"], {}).setdefault(i["verdict"], 0)
        by[i["type"]][i["verdict"]] += 1
    return {"counts": by, "latest": [{k: i.get(k) for k in ("id", "type", "title", "verdict")} | {"by": i["provenance"]["added_by"], "agent": i["provenance"].get("agent")} for i in items[-25:]]}


def picture_agreement(a, b):
    """Agreement of two annotations of one folio on the closed picture vocabulary (vs/pictures.py)."""
    from .pictures import agreement
    return agreement(a, b)
