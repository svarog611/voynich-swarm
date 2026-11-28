"""Test calibration: the runs that show the test itself works, before anyone relies on it.

    python3 -m vs season --split heldout

Two things only, both about the TEST, not about the manuscript:
  controls  genuine text in each candidate language, enciphered with random keys: the true
            key must pass, and the pinned search must find a passing key (else a grid cell
            in that language can never be ruled out: "no power").
  redteam   decoders built to look fluent (every word -> one Latin word; a hashed real
            word) must fail.
Findings about the manuscript are never produced here: they come only from agents' pull
requests, through vs/score.py, onto the record (registry/items.jsonl).
"""
import json, os, time
from . import path, corpus, judge, controls, search, refcorpus

def _langs():
    from . import refcorpus
    base = ["latin", "italian", "spanish", "french", "catalan", "german", "english", "czech", "hebrew"]
    # constructed languages (conlang_*) are pinned as references for the hypothesis matrix (P-J-004),
    # not as candidate plaintexts: no controls, no grid cells, no first-key steps for them
    return base + [l for l in refcorpus.languages() if l not in base and not l.startswith("conlang_")]


LANGS = _langs()
FAMILIES = ["substitution", "verbose", "nomenclator", "table-cipher", "abbreviation", "syllabic", "nulls", "letter-words"]
CONTROL_FOR = {"substitution": "homophonic", "verbose": "verbose", "nomenclator": "nomenclator", "table-cipher": "table-cipher", "abbreviation": "abbreviation", "syllabic": "syllabic", "nulls": "nulls", "letter-words": "letter-words"}
GRID_FAMILIES = ["substitution", "verbose", "nomenclator", "table-cipher", "abbreviation", "syllabic", "nulls", "letter-words"]


def _save(split, name, obj):
    d = path("results", split)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)


def _short(r):
    keep = {k: r[k] for k in ("submission", "family", "target", "verdict", "failed") if k in r}
    keep["passes_without_anchors"] = r["a_compresses"]["pass"] and r["b_reads"]["pass"] and r["kill_tests"]["pass"]
    keep["savings"] = r["a_compresses"]["savings"]
    keep["killed_by"] = [n for n, v in r["kill_tests"]["decoys"].items() if v["killed"]]
    return keep


def _german_verbose_sizes(pages, true_key, log):
    """Report two additional sample sizes on disjoint synthetic training and
    evaluation pages. The grid uses only the standard one-in-seven verdict."""
    rows = []
    for residues in ((3, 4), (3, 4, 5)):
        train, evaluation = [], []
        for page in sorted(pages, key=lambda p: int(p.id[1:])):
            (evaluation if int(page.id[1:]) % 7 in residues else train).append(page)
        key = search.search_cell("german", "verbose", train, tag="control")
        planted = _short(judge.evaluate(true_key, train, evaluation, "control"))
        fitted = _short(judge.evaluate(key, train, evaluation, "control"))
        tokens = sum(len(line.words) for p in evaluation for line in p.lines)
        rows.append({"evaluation_residues_mod_7": list(residues), "tokens": tokens,
                     "true_key": planted, "search_key": fitted})
        log(f"sample size german verbose 2% noise tokens={tokens} "
            f"planted={planted['passes_without_anchors']} search={fitted['passes_without_anchors']}")
    return rows


class ControlUnavailable(ValueError):
    """The generator cannot build this family's control for the target language."""


def calibrate_cell(lang, family):
    """Measure clean and noisy controls with the same search budget used on the grid."""
    kind = CONTROL_FOR[family]
    results, recovery = {}, {}
    for noise in (0.0, controls.NOISE):
        try:
            train, evaluation, planted = controls.make(lang, kind, noise=noise)
        except (ValueError, IndexError) as ex:
            raise ControlUnavailable(str(ex)) from ex
        fitted = search.search_cell(lang, family, train, tag="control")
        if family == "letter-words":
            from .control_recovery import letter_words_recovery
            if noise == 0.0:
                clean_pages = train, evaluation
            recovery["clean" if noise == 0.0 else "noisy"] = letter_words_recovery(
                planted["key"], fitted["key"], clean_pages, (train, evaluation)
            )
        results[noise] = (_short(judge.evaluate(planted, train, evaluation, "control")),
                          _short(judge.evaluate(fitted, train, evaluation, "control")),
                          sum(len(line.words) for page in evaluation for line in page.lines))
    clean_true, clean_search, tokens = results[0.0]
    noisy_true, noisy_search, _ = results[controls.NOISE]
    row = {"language": lang, "family": family, "control": kind, "control_tokens": tokens,
            "true_key": clean_true, "true_key_passes": clean_true["passes_without_anchors"],
            "search_key": clean_search, "search_cracks_control_clean": clean_search["passes_without_anchors"],
            "noise": controls.NOISE, "true_key_passes_noisy": noisy_true["passes_without_anchors"],
            "search_cracks_control": noisy_search["passes_without_anchors"]}
    if recovery:
        row["letter_words_recovery"] = recovery
    return row


def run(split, log=print, parts=("controls", "redteam")):
    train, ev = corpus.split_pages(split)
    t0 = time.time()
    if "controls" in parts:
        rows = []
        for lang in LANGS:
            for fam in FAMILIES:
                if (fam in ("nomenclator", "table-cipher") and len(refcorpus.alphabet(lang)) > len(controls.CIPHER_GLYPHS)) or (fam == "nomenclator" and len(refcorpus.alphabet(lang)) > len(controls.CIPHER_GLYPHS) or ( fam == "abbreviation" and len(refcorpus.alphabet(lang)) > len(controls.CIPHER_GLYPHS) - len(controls.ABBREVIATION_SIGNS))):
                    log(f"control {lang:8s} {fam:12s} unsupported alphabet size")
                    continue
                if fam == "syllabic" and (not set(refcorpus.alphabet(lang)) & set("aeiou")
                                          or len(refcorpus.alphabet(lang)) >= len(controls.CIPHER_GLYPHS)):
                    log(f"control {lang:8s} {fam:12s} unsupported spelling or vowel inventory")
                    continue
                try:
                    row = calibrate_cell(lang, fam)
                except ControlUnavailable as ex:
                    log(f"control {lang:8s} {fam:12s} cannot build a control ({str(ex)[:60]})")
                    continue
                rows.append(row)
                log(f"control {lang:8s} {fam:12s} clean={row['search_cracks_control_clean']} "
                    f"noisy={row['search_cracks_control']}")
                if (lang, fam) == ("german", "verbose"):
                    ctr, cev, true = controls.make(lang, CONTROL_FOR[fam], noise=controls.NOISE)
                    row["noisy_sample_size_checks"] = _german_verbose_sizes(ctr + cev, true, log)
        _save(split, "controls", rows)
    if "redteam" in parts:
        rows = [_short(judge.evaluate(k, train, ev, split)) for lang in ("latin", "italian") for k in controls.red_team_keys(lang)]
        _save(split, "redteam", rows)
        log(f"red-team decoders: {sum(r['verdict'] == 'FAIL' for r in rows)}/{len(rows)} rejected")
    _save(split, "summary", {"split": split, "environment": judge.environment(), "pages_evaluated": len(ev),
                             "seconds": round(time.time() - t0)})
