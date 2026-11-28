"""Program R: the red team. Its only job is to break claims; a successful attack weighs the
same as a result in the ledger (on the path when the target is on the path).

Three kinds of attack, each a mechanical check, so no attack is settled by argument:

  key-attack     Target: a key that passes some of the judge. The attacker names a DECOY
                 (a pinned decoy constructor plus parameters) on which the key's output also
                 passes (b) "reads". Success: the key reads the decoy about as well as the
                 manuscript, so its fluency is not evidence. A successful decoy is proposed
                 for the next season's kill tests (it must not kill any positive control).

  fact-attack    Target: an ESTABLISHED fact that is claimed to show language or meaning.
                 The attacker supplies SYNTHETIC TEXT (data, not code: a list of lines of
                 words, made without the held-out pages) plus a one-paragraph method. The
                 scorer cuts it into the evaluation pages' shapes and reruns the fact.
                 Success: the fact holds on the synthetic text too, so it needs no language.
                 Guard: the text must not copy the manuscript's phrases (no run of 4 consecutive
                 manuscript words). Using the manuscript's own words is allowed, as period
                 table procedures would; the share is reported.

  claim-attack   Target: a published or submitted decipherment claim. The attacker submits
                 the claim as a key (family + key) and the judge rules. A FAIL is a
                 successful attack and goes in the refutation log with the claimant named.

Attack experiments are experiments/<step>/<handle>/ with experiment.json kind
"attack" plus attack.json; see join.txt.
"""
import json, random
from . import corpus, facts, judge, ivtff
from .ivtff import Page, Line

DECOYS = {
    "word_shuffle": "real evaluation pages with words shuffled within each page (params: seed)",
    "glyph_shuffle": "real words with glyphs shuffled within each word (params: seed)",
    "random_glyphs": "random glyphs by training frequency, word lengths kept (params: seed)",
    "reversed_words": "every real word spelled backwards (params: none)",
    "other_language": "a reference language's control text enciphered into EVA (params: language, seed)",
    "cross_section": "pages of one section relabelled as another (params: from, to)",
}


def build_decoy(kind, params, eval_pages, base, seg):
    rng = random.Random(params.get("seed", 1))
    shape = [(p, ln) for p in eval_pages for ln in p.lines if ln.words]
    toks = [w for _p, ln in shape for w in ln.words]
    if kind == "word_shuffle":
        out = []
        for p in eval_pages:
            ws = [w for ln in p.lines for w in ln.words]
            rng.shuffle(ws)
            it = iter(ws)
            out += [[next(it) for _ in ln.words] for ln in p.lines if ln.words]
        return out
    if kind == "glyph_shuffle":
        new = []
        for w in toks:
            g = seg(w)
            rng.shuffle(g)
            new.append("".join(g))
    elif kind == "random_glyphs":
        gl = [g for g in base.glyph_freq if g != "<stop>"]
        gw = [base.glyph_freq[g] for g in gl]
        new = ["".join(rng.choices(gl, gw, k=len(seg(w)))) for w in toks]
    elif kind == "reversed_words":
        new = ["".join(reversed(seg(w))) for w in toks]
    elif kind == "other_language":
        from . import refcorpus
        ow = " ".join(refcorpus.control_text(params["language"])).split()
        glyphs = [g for g, _ in base.glyph_freq.most_common() if g != "<stop>" and not ivtff.uncertain(g)]
        rng.shuffle(glyphs)
        letters = sorted(set("".join(ow)))
        cip = {l: glyphs[i % len(glyphs)] for i, l in enumerate(letters)}
        new = ["".join(cip[c] for c in w) for w in (ow * (1 + len(toks) // max(len(ow), 1)))[: len(toks)]]
    elif kind == "cross_section":
        src = [p for p in corpus.pages() if p.section == params["from"]]
        new = [w for p in src for ln in p.lines for w in ln.words][: len(toks)]
        new += toks[len(new):]
    else:
        raise SystemExit(f"unknown decoy {kind!r}; pinned decoys: {sorted(DECOYS)}")
    it = iter(new)
    return [[next(it) for _ in ln.words] for _p, ln in shape]


def key_attack(target_key, decoy, params, split):
    train, ev = corpus.split_pages(split)
    lang = target_key["target"]["language"]
    seg = judge.seg_fn(target_key.get("segmentation", "eva-basic"))
    base = judge.Baseline(train, seg)
    dec = judge.Decoder(target_key, lang)
    cal = judge.calibration(lang)
    lines = build_decoy(decoy, params, ev, base, seg)
    real = judge.text_metrics(lang, dec.words([w for p in ev for ln in p.lines for w in ln.words]))
    fake = judge.text_metrics(lang, dec.words([w for l in lines for w in l]))
    reads = fake["H"] <= cal["H_max"] and fake["cov"] >= cal["cov_min"]
    close = fake["H"] <= real["H"] + judge.PARAMS["kill_char_gap"] and fake["cov"] >= real["cov"] - judge.PARAMS["kill_cov_gap"]
    return {"attack": "key", "target": target_key.get("id"), "decoy": decoy, "params": params,
            "real": {k: round(v, 4) for k, v in real.items() if k in ("H", "cov")},
            "decoy_output": {k: round(v, 4) for k, v in fake.items() if k in ("H", "cov")},
            "success": bool(reads or close),
            "why": "decoy read about as fluently as the manuscript" if (reads or close) else "the key does not read this decoy"}


def _manuscript_4grams():
    out = set()
    for p in corpus.pages():
        for ln in p.lines:
            w = ln.words
            out.update(tuple(w[i:i + 4]) for i in range(len(w) - 3))
    return out


def _count_openers(pages):
    """Lines that open a paragraph, counted as vs/dsl.py tokens() marks para_first_line."""
    n = 0
    for p in pages:
        first_line = True
        for ln in p.lines:
            if not ln.words:
                continue
            n += ln.unit == "P" and (ln.para_start or first_line)
            first_line = ln.para_end if ln.unit == "P" else first_line
    return n


def fact_attack(fact_id, lines, split, fact_item=None, cut="tokens", first_lines=None):
    """lines: list of lists of words (the attacker's synthetic text). fact_item: the
    registry item for the target fact (its detail holds the metric and params).

    cut: how the synthetic text is laid onto the evaluation pages.
      "tokens" (default): one token stream cut to the pages' line lengths. The attacker's line
               breaks are lost, so no generator can know where a line ends: a positional fact
               (line-initial, line-final) survives every such attack by construction (R-001).
      "lines": each evaluation line takes the next synthetic line whole (its length is the
               attacker's). This lets a procedure that sees the line end, as any scribe or
               table method writing on a page does, be tested against positional facts.
               Page and line metadata (section, language, unit, paragraph marks) stay the
               evaluation pages'.
      "book":  as "lines", but the text is laid onto EVERY page of the manuscript in order and the
               fact is then measured on the evaluation pages only, each holding the lines that fell
               on it. With "lines" the text runs end to end over a sample of pages, so two evaluation
               pages that stand fifteen pages apart receive text written a page or two apart, and a
               short memory passes for long-range structure (P-G-007: a drift-free text passes F-011
               that way and fails it once the book is sampled). "book" asks the procedure to write
               the whole book and reads the sampled pages of it. The line counts it needs are the whole
               manuscript's, which are public; nothing about the evaluation pages is revealed.

    first_lines (optional, only with cut "lines"): lines the procedure wrote as the first line of a
               paragraph. Each evaluation line that opens a paragraph (the measurement language's
               para_first_line) takes the next of these instead of the next of `lines`. A writer
               knows when a paragraph begins, as it knows when a line does; without this no text
               can meet a fact about paragraph first lines except by making every line alike."""
    lines = [[str(w) for w in l if str(w)] for l in lines if l]
    first_lines = [[str(w) for w in l if str(w)] for l in (first_lines or []) if l]
    toks = [w for l in lines + first_lines for w in l]
    ms4 = _manuscript_4grams()
    copied = sum(tuple(l[i:i + 4]) in ms4 for l in lines + first_lines for i in range(len(l) - 3))
    types = {w for p in corpus.pages() for w in p.words()}
    shared = sum(w in types for w in toks) / max(len(toks), 1)
    train, ev = corpus.split_pages(split)
    need = sum(len(ln.words) for p in ev for ln in p.lines)
    ev_ids = {p.id for p in ev}
    target = list(corpus.pages()) if cut == "book" else ev      # the pages the text is laid onto, in order
    guard = []
    if copied:
        guard.append(f"{copied} four-word runs copied from the manuscript")
    marked = sum(1 for w in toks if "?" in w or "*" in w)
    if marked:
        guard.append(f"{marked} tokens carry uncertainty markers (? or *): synthetic text may not use them "
                     "(they are a transcription convention, and metrics skip them; found by the red team, R-003)")
    if cut not in ("tokens", "lines", "book"):
        guard.append(f"cut must be 'tokens', 'lines' or 'book', got {cut!r}")
    elif cut == "book":
        openers = _count_openers(target)
        total = sum(1 for p in target for ln in p.lines if ln.words)
        need_plain = total - openers if first_lines else total
        if len(lines) < need_plain:
            guard.append(f"cut 'book' needs at least {need_plain} lines to fill every page of the manuscript, got {len(lines)}")
        if first_lines and len(first_lines) < openers:
            guard.append(f"cut 'book' needs at least {openers} first_lines (the manuscript's paragraph-opening lines), got {len(first_lines)}")
    elif cut == "lines":
        need_lines = sum(1 for p in ev for ln in p.lines if ln.words)
        if len(lines) < need_lines:
            guard.append(f"cut 'lines' needs at least {need_lines} lines to fill the evaluation pages, got {len(lines)}")
        if first_lines and len(first_lines) < need_lines:
            guard.append(f"first_lines needs at least {need_lines} lines (any line may open a paragraph), got {len(first_lines)}")
    if first_lines and cut not in ("lines", "book"):
        guard.append("first_lines needs cut 'lines' or 'book'")
    elif len(toks) < need:
        guard.append(f"needs at least {need} tokens to fill the evaluation pages, got {len(toks)}")
    if guard:
        return {"attack": "fact", "target": fact_id, "success": False, "refused": guard}
    it = iter(toks)
    lit = iter(lines)
    fit = iter(first_lines)
    fake = []
    for p in target:
        q = Page(p.id, p.order, dict(p.meta))
        first_line = True                      # as vs/dsl.py tokens(): para_first_line
        for ln in p.lines:
            opens = bool(ln.words) and ln.unit == "P" and (ln.para_start or first_line)
            if ln.words:
                first_line = ln.para_end if ln.unit == "P" else first_line
            if cut in ("lines", "book") and first_lines and opens:
                ws = list(next(fit))
            elif cut in ("lines", "book"):
                ws = list(next(lit)) if ln.words else []
            else:
                ws = [next(it) for _ in ln.words]
            q.lines.append(Line(ln.id, ln.page, ln.n, ln.unit, ln.subtype, ws, ln.para_start, ln.para_end))
        if p.id in ev_ids:                     # with cut "book" the other pages only take up their share of the text
            fake.append(q)
    if fact_item is None:
        from . import registry
        fact_item = registry.get(fact_id)
    if not fact_item or fact_item["type"] != "fact":
        return {"attack": "fact", "target": fact_id, "success": False, "refused": [f"{fact_id} is not a fact on the record"]}
    if "spec" in fact_item["detail"]:
        from . import dsl
        spec = fact_item["detail"]["spec"]
        real = dsl.evaluate(spec, ev, train)
        try:
            got = dsl.evaluate(spec, fake, train)
        except dsl.SpecError as ex:
            # The fact is defined on the manuscript (it was measured there) but not on this text: too
            # few of the words it counts, or nothing for its null to shuffle. That is the synthetic
            # text failing to reproduce the fact, not a fault of the scorer.
            return {"attack": "fact", "target": fact_id, "real": {"pass": real["pass"], "value": real["value"]},
                    "synthetic": {"pass": False, "value": None, "undefined": str(ex)[:200]},
                    "shared_type_share": round(shared, 3), "cut": cut, "success": False,
                    "why": "the fact cannot be measured on the synthetic text, so the text does not reproduce it"}
    else:
        m, prm = fact_item["detail"]["metric"], fact_item["detail"]["params"]
        real = facts.METRICS[m](train, ev, **prm)
        got = facts.METRICS[m](train, fake, **prm)
    det = got.get("detail") or {}
    if "rate_in" in det and "n_in" in det and round(det["rate_in"] * det["n_in"]) < 10:
        return {"attack": "fact", "target": fact_id, "success": False,
                "refused": ["positional attack with fewer than 10 hits: too few to reproduce a fact (P-J-001)"]}
    return {"attack": "fact", "target": fact_id, "real": {"pass": real["pass"], "value": real["value"]},
            "synthetic": {"pass": got["pass"], "value": got["value"]}, "shared_type_share": round(shared, 3), "cut": cut,
            "success": bool(real["pass"] and got["pass"]),
            "why": "the fact also holds on text made without language" if (real["pass"] and got["pass"]) else "the synthetic text does not reproduce the fact"}


def run_attack(spec, split):
    kind = spec["attack"]
    if kind == "key":
        tk = spec["target_key"]
        if isinstance(tk, str) and not tk.endswith(".json"):
            from . import registry   # a record id: attack the key exactly as it was submitted
            it = registry.get(tk)
            if not it or "key.json" not in (it.get("submitted") or {}):
                return {"attack": "key", "target": tk, "success": False, "refused": [f"{tk} is not a key on the record"]}
            key = it["submitted"]["key.json"]
        else:
            key = json.load(open(tk)) if isinstance(tk, str) else tk
        return key_attack(key, spec["decoy"], spec.get("params", {}), split)
    if kind == "fact":
        return fact_attack(spec["target_fact"], spec["lines"], split, spec.get("target_fact_item"),
                           cut=spec.get("cut", "tokens"), first_lines=spec.get("first_lines"))
    if kind == "generator":
        return generator_test(spec["lines"], split, cut=spec.get("cut", "lines"), facts_on_record=spec.get("facts_on_record"),
                              first_lines=spec.get("first_lines"))
    if kind == "claim":
        train, ev = corpus.split_pages(split)
        r = judge.evaluate(spec["key"], train, ev, split)
        return {"attack": "claim", "target": spec["key"].get("id"), "claimant": spec.get("claimant"),
                "verdict": r["verdict"], "failed": r["failed"], "success": r["verdict"] == "FAIL"}
    raise SystemExit("attack must be key, fact, generator or claim")


def best_generator(attacks=None):
    """The meaningless whole-book generator on the record that reproduces the most established
    facts (ties: the earlier one), as its registry item, or None. Invented-language and
    enciphered-language tests are not generators of meaningless text and are left out."""
    from . import registry
    attacks = registry.read("attack") if attacks is None else attacks
    facts = {i["id"] for i in registry.read("fact") if i["verdict"] == "ESTABLISHED"}
    best = None
    for a in attacks:
        d, sub = a.get("detail") or {}, (a.get("submitted") or {}).get("attack.json") or {}
        if d.get("attack") != "generator" or d.get("hypothesis") in ("invented_language", "enciphered_language") or not sub.get("lines"):
            continue
        n = sum(1 for k, v in (d.get("per_fact") or {}).items() if v and k in facts)
        if best is None or n > best[0]:
            best = (n, a)
    return best[1] if best else None


def against_best_generator(fact_item, split):
    """Run the best generator on the record against one new fact, as an attack on it would be run.
    Only the outcome is returned (no hidden values): {"attack": id, "reproduced": bool} or None."""
    g = best_generator()
    if g is None:
        return None
    sub = g["submitted"]["attack.json"]
    r = fact_attack(fact_item["id"], sub["lines"], split, fact_item, cut=sub.get("cut", "lines"),
                    first_lines=sub.get("first_lines"))
    return {"attack": g["id"], "reproduced": bool(r.get("success")) and not r.get("refused")}


def generator_test(lines, split, cut="lines", facts_on_record=None, first_lines=None):
    """Program G's scoreboard: one body of synthetic text from one procedure, run against EVERY
    established fact at once. The score is how many it reproduces. A procedure that reproduces
    them all is strong evidence the text needs no language; facts no general procedure can
    reproduce are the strongest public evidence of structure. Only pass/fail per fact goes on
    the record (no hidden values)."""
    if facts_on_record is None:
        from . import registry
        facts_on_record = [i for i in registry.read("fact") if i["verdict"] == "ESTABLISHED"]
    per = {}
    for f in facts_on_record:
        r = fact_attack(f["id"], lines, split, f, cut=cut, first_lines=first_lines)
        if r.get("refused"):
            text_problem = [g for g in r["refused"] if not g.startswith("positional attack")]
            if text_problem:
                return {"attack": "generator", "target": "*", "success": False, "refused": text_problem}
            per[f["id"]] = False   # too few hits to count: not reproduced
        else:
            per[f["id"]] = bool(r["success"])
    k = sum(per.values())
    return {"attack": "generator", "target": "*", "cut": cut, "reproduced": k, "of": len(per), "per_fact": per,
            "success": bool(per) and k == len(per),
            "why": f"the text reproduces {k} of {len(per)} established facts"}
