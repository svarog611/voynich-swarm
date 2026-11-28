"""Measurement language: a fact claim is a declarative spec interpreted by this trusted code.
No agent code is ever executed by the scorer. A spec is JSON:

  {"scope":     {"hand": "1"|..., "language": "A"|"B", "section": "H"|...},        (optional)
   "select":    {"unit": ["P"], "position": "line_final", "certain": true},       (optional)
   "statistic": {...one of the forms below...},
   "compare":   [ ...one to four tests, all must pass... ]}

Selections may also set language to A or B to compare the two dialects within a shared scope.
positions: any | line_initial | line_final | mid_line | para_initial (first word of a paragraph)
           | para_first_line (any word on a paragraph's first line)
           | not_para_first_line (paragraph text that is not on a first line; never labels or rings)
           | para_first_line_rest / not_para_first_line_rest (the same two, leaving out the first word of
             each line: a first-line pattern that is more than the paragraph's opening word; found by the attack on F-008)
features:  "word" | {"glyph_at": i} (eva-basic glyphs; negative counts from the end)
           | {"prefix": n} | {"suffix": n} (in glyphs, 1..8) | "length" | "glyph" (every glyph of the word)

statistics:
  {"rate": {"feature": F, "in": [values]}}                 share of selected tokens whose feature is in values
  {"ratio": {"feature": F, "in": [values], "of": SELECT, "to": SELECT}}   rate under one selection over another
  {"entropy": F}                                           plug-in entropy of the feature (bits)
  {"cond_entropy": F}                                      H(next word's F | previous word's F), adjacent words in a line
  {"glyph_cond_entropy": k}                                H(glyph | previous k glyphs), k = 1..3, inside lines with word
                                                           boundaries as a symbol (k=1 is Bennett's second-order entropy)
  {"adjacent_mi": F}                                       adjacent mutual information minus the same after shuffling
                                                           each line (so page and section vocabulary cancel)
  {"boundary_class_rate": true}                         Fraction of within-line eligible pairs whose left ending is end-preferring
                                                           and right onset start-preferring. Classes are fitted ONLY on training
                                                           pages with a fixed 2:1 rule and Parisel pool; requires training pages.
  {"edge_mi": true}                                        MI between a word's last glyph and the next word's first glyph
  {"edge_length_cmi": true}                               conditional MI of those edges given both exact EVA-basic word
                                                           lengths, weighted by adjacent-pair count (bits). Certain P
                                                           words only, position any; gaps remain barriers. At least 30 pairs.
                                                           Its null shuffle_within_line_length permutes intact certain P
                                                           words inside (original line, exact glyph length) cells, preserving
                                                           the length sequence; at least 30 potentially movable pairs needed.
  {"edge_mi_break": true}                                  edge MI within lines minus edge MI across line breaks (a line's
                                                           last word with the next line's first word, same page); each
                                                           shuffle-corrected, within-line pairs subsampled to the
                                                           across-line count (10 fixed draws) so both carry equal bias
  {"paragraph_gap_corr": {"symbol": "o"}}                 pooled endpoint-centered correlation of successive raw-character
                                                           occurrence gaps at lags 1..5, averaged; requires complete,
                                                           certain P lines containing only the eligible EVA alphabet and
                                                           at least 25 contributing pairs at every lag. It accepts
                                                           shuffle_within_paragraph and shuffle_within_line, which permute
                                                           intact words inside eligible runs or original lines.
  {"line_repeat_excess": true}                              share of words repeated on the previous numbered line minus the share
                                                           repeated four lines back, within a page, excluding one-character words
  {"line_near_excess": {"min_length": L}}                  the same comparison for similar words (Timm 2014): share of words of
                                                           length >= L (default 4) with a word at edit distance exactly 1 on the
                                                           previous numbered line, minus the share with one four lines back
  {"repeat_ngram": n}                                      share of word n-grams (n<=6) that occur at least twice
  {"repeated_pair_types": true}                            share of distinct directed within-line word pairs appearing
                                                           at least twice; certain P words, with gaps as barriers.
                                                           shuffle_within_line permutes within each certain run.
  {"hapax_share": true} | {"zipf_slope": ranks} | {"mean_length": true}
  {"directional_pairs": {"window": W, "drop_top": K, "excess": T}}
                                                           Fraction of observed unordered, distinct word-type pairs
                                                           whose forward/reverse occurrence counts differ by more than T.
                                                           Count pairs up to W positions apart after removing the K
                                                           most frequent types (including all types tied at the cutoff).
                                                           Count only within original contiguous selected line runs;
                                                           stop-word removal may close gaps within a run, but never
                                                           bridges uncertain/unselected tokens or line boundaries.
                                                           Requires 30 retained tokens and 30 observed pair types.
                                                           This fixes one convention for ambiguities in the published
                                                           denominator and filtering; it is not an exact paper replay.
  {"filtered_vocab": {"measure": M, "drop_top": K}}          M is type_token or singleton_token: distinct types, or
                                                           singleton types, divided by retained token count. Frequency
                                                           filtering uses the same explicit K and tie rule as above.
                                                           These are distinct definitions of 'unique word ratio'.
  {"near_copy": {"window": k, "min_length": L}}            share of words of length >= L within edit distance exactly 1
                                                           of one of the previous k (1..50) on the page
  {"page_similarity": {"within": d, "min_words": m}}       median cosine of word-count vectors of two pages at most d pages
                                                           apart in manuscript order (d 1..50), minus the median for pages
                                                           farther apart; pages with fewer than m (default 40) selected words
                                                           are left out. Distance is counted in the manuscript's page order,
                                                           so it stays meaningful on a sample of pages
  {"page_group_similarity": {"by": B, "min_words": m}}     median cosine of word-count vectors of two pages that share B
                                                           (section | language | hand) minus the median for two pages that
                                                           differ in it; pages with fewer than m (default 40) selected words
                                                           or without that label are left out. Every page stays whole, so page-level reuse cancels;
                                                           test it against shuffle_page_labels and rotate_page_labels
  {"page_reuse": {"min_length": L}}                        share of words of length >= L already written earlier on the page
  {"matched_rightwardness": {"contrast": C, "min_line": a, "max_line": b}}
                                                           Mean relative interior position of the second member of a
                                                           matched-word contrast minus the first, pooled by token.
                                                           C is sh/ch, qo/o or k/t; line lengths are original token counts.
                                                           Matching uses certain paragraph tokens in the selected sample.
                                                           At least 30 tokens per side and 5 pages are required.
                                                           This is a pooled contrast: page-specific composition can
                                                           create a difference even with no within-page difference.
  {"picture_label": {"test": T, "min_objects": n}}         labels against AGREED picture annotations (registry type picture;
                                                           one annotation per folio, the best-agreeing). Objects must be
                                                           independently matched without using labels. Two visually agreeing
                                                           seats must support the same label line (unit L) on the scored pages.
                                                           Its words are the label. T is
       same_object_label    pairs of objects on different folios that look alike (same kind, and at least 80% of at
                            least two shared features equal) have labels identical or one glyph apart, minus the
                            same rate over all other cross-folio pairs
       feature_label_share  pairs of objects of the same kind sharing at least one feature value have labels that
                            share a first or last glyph pair (eva-basic), minus the rate for pairs sharing none
       label_in_page_text   share of labels with a word (3+ glyphs) in their own page's running text, minus the
                            mean share for the other scored pages' running text
                            Two matching explicit loci suffice. Otherwise both seats must independently locate the
                            same nearest label in the same ring within 20 degrees of their clock positions, with
                            no contrary explicit locus. Ties, competing assignments and disputed links are excluded.
                            ZL3b gives clock positions in <!hh:mm> comments. Rings are the label runs between circular text lines, outermost first;
                            a run before the first circular line (figures outside the circle) is not a ring.
                            Layout fields (ring, order, angle, clock, position, label, note, count, group) are not
                            looks and are ignored. n (default 20, at least 10) labelled objects are required, and 5
                            pairs in each group compared; fewer is undefined (refused). Test it against
                            shuffle_picture_labels.
  {"context_mi": {"feature": F, "by": B}}                  MI between the feature and where the word stands; B is
                                                           page | section | unit | line_position (initial/mid/final).
                                                           Plug-in MI is biased upward: only meaningful against a null

compare (one to four; all must pass; a spec with none is refused):
  {"op": ">=" | "<=" | ">" | "<", "value": x}              x finite
  {"page_bootstrap": "lower" | "upper", "value": x}         matched_rightwardness only: 95% two-sided percentile
                                                           interval from 399 whole-page resamples, fixed seed. Lower
                                                           must exceed x, or upper must be below x, respectively.
                                                           Requires at least 10 pages with matched tokens; coverage is
                                                           not guaranteed for sparse or unrepresentative pages.
                                                           The matched vocabulary is rebuilt on each resample.
  {"reference": "below_all" | "above_all" | "inside_range", "margin": m}     m >= 0
       the same statistic on every reference language, on that language's words poured into the exact layout of
       the selected tokens (same pages, lines, gaps and count), so size and shape are matched. Languages with
       fewer words than the selection are left out; at least three must remain. Only for statistics whose
       features carry over to another script (not rate, ratio or context_mi).
  {"null": "shuffle_within_line" | "shuffle_within_line_length" | "shuffle_within_page" | "shuffle_within_section" | "shuffle_across_pages"
           | "shuffle_labels" | "shuffle_labels_within_page" | "shuffle_page_labels" | "rotate_page_labels",
           | "shuffle_picture_labels",
   "perms": N (<= 400), "p": alpha (<= 0.05), "direction": "greater" | "less"}
       the statistic must be extreme against the null (one-sided permutation p-value, ties count against).
       shuffle_labels reassigns which words of a section are the labels (labels trade places with words
       of running text of the same section; all other words stay where they are, so word order elsewhere is
       untouched). shuffle_labels_within_page does the same inside each page (stricter; labels on pages
       without running text cannot move).
       shuffle_page_labels keeps every page's text whole and deals the pages' labels (section, language and
       hand, together) among the pages. rotate_page_labels is stricter: the labels keep their order and their
       runs and are shifted along the page order by every possible offset (perms is ignored: all n-1 shifts
       of the n pages are tried, so p cannot be below 1/n), which a text that merely changes slowly from
       front to back does not beat. Both leave the words where they are: they test only statistics that read
       the labels (page_group_similarity, context_mi by section).
       shuffle_picture_labels keeps every label line's words together and deals them among the label lines that
       agreed picture objects point at, so labels move between objects (and pages) while running text stays put.

Rules that keep a pass from being hollow (each has a test in tests/test_dsl.py):
  - unknown keys anywhere are refused (a misspelt "scope" or "select" used to be silently ignored);
  - at least one comparison; thresholds, margins and alpha must be finite, margins >= 0, alpha <= 0.05, and
    perms large enough to reach alpha;
  - a statistic that is undefined on the selection (fewer than 30 tokens, or fewer than 30 adjacent pairs
    / n-grams / long words for the statistics that count those) is refused, not reported as 0;
  - a value that is not finite (a ratio with an empty denominator) never passes;
  - adjacency means adjacent in the line: words either side of a dropped (uncertain or unselected) word
    are not a pair;
  - a null that cannot move the statistic (all permuted values equal the observed one) is refused.

Bounded cost by construction: n-grams <= 6, windows <= 50, permutations <= 400 per null, comparisons <= 4.
"""
import math, random
from collections import Counter
from functools import lru_cache
from math import fsum, sqrt
from statistics import mean
from . import ivtff

UNCERTAIN = ivtff.uncertain
MAX_COMPARE, MAX_PERMS, MAX_ALPHA = 4, 400, 0.05
MIN_TOKENS = 30          # selected tokens, and counted units (pairs, n-grams, long words) where a statistic counts them
UNITS = {"P", "L", "C", "R"}
POSITIONS = {"any", "line_initial", "line_final", "mid_line", "para_initial", "para_first_line", "not_para_first_line",
             "para_first_line_rest", "not_para_first_line_rest"}
NAN = float("nan")


@lru_cache(maxsize=200_000)
def SEG(w):
    return tuple(ivtff.segment(w, "eva-basic"))


class SpecError(ValueError):
    pass


def _keys(d, allowed, what, required=()):
    if not isinstance(d, dict):
        raise SpecError(f"{what} must be an object")
    extra = set(d) - set(allowed)
    if extra:
        raise SpecError(f"{what}: unknown key(s) {sorted(extra)}; allowed: {sorted(allowed)}")
    for k in required:
        if k not in d:
            raise SpecError(f"{what} needs {k!r}")


def _num(x, what):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise SpecError(f"{what} must be a finite number")
    return float(x)


def _int(x, lo, hi, what):
    if isinstance(x, bool) or not isinstance(x, int) or not lo <= x <= hi:
        raise SpecError(f"{what} must be an integer in {lo}..{hi}")
    return x


# ------------------------------------------------------------- tokens

def tokens(pages, scope=None):
    """One record per word, with its layout context."""
    scope = scope or {}
    out = []
    for pi, p in enumerate(pages):
        if scope.get("hand") and str(p.hand) != str(scope["hand"]):
            continue
        if scope.get("language") and p.language != scope["language"]:
            continue
        if scope.get("section") and p.section != scope["section"]:
            continue
        first_line = True
        for ln in p.lines:
            if not ln.words:
                continue
            pfl = ln.unit == "P" and (ln.para_start or first_line)
            n = len(ln.words)
            for k, w in enumerate(ln.words):
                out.append({"w": w, "page": pi, "section": p.section, "line": (pi, ln.n), "k": k, "n": n,
                            "language": p.language, "hand": str(p.hand), "unit": ln.unit, "pfl": pfl, "pinit": pfl and k == 0,
                            "order": p.order, "lid": ln.id, "folio": p.id})
            first_line = ln.para_end if ln.unit == "P" else first_line
    return out


def check_select(sel, what="select"):
    sel = sel or {}
    _keys(sel, {"unit", "position", "certain", "language"}, what)
    if "language" in sel and sel["language"] not in ("A", "B"):
        raise SpecError(f"{what}.language must be A or B")
    u = sel.get("unit") or []
    if not isinstance(u, list) or set(u) - UNITS:
        raise SpecError(f"{what}.unit is a list from {sorted(UNITS)}")
    if sel.get("position", "any") not in POSITIONS:
        raise SpecError(f"unknown position {sel.get('position')!r}")
    if not isinstance(sel.get("certain", True), bool):
        raise SpecError(f"{what}.certain is true or false")
    return sel


def select(toks, sel):
    sel = check_select(sel)
    units = set(sel.get("unit") or [])
    pos = sel.get("position", "any")
    certain = sel.get("certain", True)
    language = sel.get("language")
    keep = []
    for t in toks:
        if language and t["language"] != language:
            continue
        if units and t["unit"] not in units:
            continue
        if certain and UNCERTAIN(t["w"]):
            continue
        ok = {"any": True, "line_initial": t["k"] == 0, "line_final": t["k"] == t["n"] - 1,
              "mid_line": 0 < t["k"] < t["n"] - 1, "para_initial": t["pinit"], "para_first_line": t["pfl"],
              "not_para_first_line": t["unit"] == "P" and not t["pfl"],
              "para_first_line_rest": t["pfl"] and t["k"] > 0,
              "not_para_first_line_rest": t["unit"] == "P" and not t["pfl"] and t["k"] > 0}[pos]
        if ok:
            keep.append(t)
    return keep


def check_feature(f):
    if f in ("word", "length", "glyph"):
        return
    if isinstance(f, dict) and len(f) == 1:
        (k, v), = f.items()
        if k == "glyph_at":
            _int(v, -12, 11, "glyph_at")
            return
        if k in ("prefix", "suffix"):
            _int(v, 1, 8, k)
            return
    raise SpecError(f"unknown feature {f!r}")


def feature(f, w):
    if f == "word":
        return [w]
    if f == "length":
        return [len(SEG(w))]
    if f == "glyph":
        return list(SEG(w))
    if isinstance(f, dict):
        g = SEG(w)
        if "glyph_at" in f:
            i = int(f["glyph_at"])
            return [g[i]] if -len(g) <= i < len(g) else []
        if "prefix" in f:
            return ["".join(g[:int(f["prefix"])])]
        if "suffix" in f:
            return ["".join(g[-int(f["suffix"]):])]
    raise SpecError(f"unknown feature {f!r}")


def lines_of(toks):
    by = {}
    for t in toks:
        by.setdefault(t["line"], []).append(t)
    return [sorted(v, key=lambda t: t["k"]) for _k, v in sorted(by.items())]


def runs_of(toks):
    """Maximal runs of words that are truly adjacent in a line (a dropped word breaks the run)."""
    out = []
    for ln in lines_of(toks):
        run = [ln[0]]
        for t in ln[1:]:
            if t["k"] == run[-1]["k"] + 1:
                run.append(t)
            else:
                out.append(run)
                run = [t]
        out.append(run)
    return out


# ------------------------------------------------------------- statistics

def _entropy(c):
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0.0


def _mi(pairs):
    pc, a, b = Counter(pairs), Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    n = sum(pc.values())
    return sum(v / n * math.log2(v * n / (a[x] * b[y])) for (x, y), v in pc.items()) if n else 0.0


def _edge_break(toks, draws=10):
    """Edge-glyph MI within lines minus across line breaks, each corrected by its shuffled value."""
    edge = lambda x, y: (SEG(x["w"])[-1], SEG(y["w"])[0])
    within = [edge(x, y) for r in runs_of(toks) for x, y in zip(r, r[1:])]
    lns = lines_of(toks)
    across = [edge(a[-1], b[0]) for a, b in zip(lns, lns[1:])
              if a[-1]["line"][0] == b[0]["line"][0] and a[-1]["k"] == a[-1]["n"] - 1 and b[0]["k"] == 0]
    n = len(across)
    if n < MIN_TOKENS or len(within) < n:
        return NAN

    def corrected(pairs, rng):
        ys = [y for _x, y in pairs]
        rng.shuffle(ys)
        return _mi(pairs) - _mi(list(zip([x for x, _y in pairs], ys)))

    rng = random.Random(1)
    w = sum(corrected(rng.sample(within, n), rng) for _ in range(draws)) / draws
    a = sum(corrected(list(across), rng) for _ in range(draws)) / draws
    return w - a


def _page_similarity(toks, within, min_words):
    """Median cosine of word counts for pages at most `within` apart in manuscript order, minus farther pairs."""
    pages = {}
    for t in toks:
        pages.setdefault(t["order"], Counter())[t["w"]] += 1
    vec = sorted((o, c, math.sqrt(sum(v * v for v in c.values()))) for o, c in pages.items()
                 if sum(c.values()) >= min_words)
    near, far = [], []
    for i, (o1, c1, n1) in enumerate(vec):
        for o2, c2, n2 in vec[i + 1:]:
            cos = sum(v * c2[w] for w, v in c1.items() if w in c2) / (n1 * n2)
            (near if o2 - o1 <= within else far).append(cos)
    if len(near) < MIN_TOKENS or len(far) < MIN_TOKENS:
        return NAN
    med = lambda xs: (lambda s: (s[(len(s) - 1) // 2] + s[len(s) // 2]) / 2)(sorted(xs))
    return med(near) - med(far)


def _page_group_similarity(toks, by, min_words):
    """Median cosine of word counts for two pages with the same label, minus pages with different labels."""
    pages, label = {}, {}
    for t in toks:
        pages.setdefault(t["page"], Counter())[t["w"]] += 1
        label[t["page"]] = t[by]
    vec = sorted((pg, c, math.sqrt(sum(v * v for v in c.values()))) for pg, c in pages.items()
                 if sum(c.values()) >= min_words and label[pg] not in (None, "", "None"))   # unlabelled pages are left out
    same, diff = [], []
    for i, (p1, c1, n1) in enumerate(vec):
        for p2, c2, n2 in vec[i + 1:]:
            cos = sum(v * c2[w] for w, v in c1.items() if w in c2) / (n1 * n2)
            (same if label[p1] == label[p2] else diff).append(cos)
    if len(same) < MIN_TOKENS or len(diff) < MIN_TOKENS:
        return NAN
    med = lambda xs: (lambda s: (s[(len(s) - 1) // 2] + s[len(s) // 2]) / 2)(sorted(xs))
    return med(same) - med(diff)


def _edit1(a, b):
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def _context(t, by):
    if by == "line_position":
        return "initial" if t["k"] == 0 else "final" if t["k"] == t["n"] - 1 else "mid"
    return t[by]


def _matched_rightwardness(toks, arg, with_counts=False):
    """Pooled second-minus-first relative positions, with a sample-derived matched vocabulary.

    Prefix contrasts match the remaining spelling exactly. For k/t, a counterpart must
    differ at exactly one k/t character; a word with multiple counterparts is counted once.
    Original line indices and lengths are retained after selection and uncertainty filtering.
    """
    ts = [t for t in toks if t['unit'] == 'P' and not UNCERTAIN(t['w']) and
          arg['min_line'] <= t['n'] <= arg['max_line'] and 0 < t['k'] < t['n'] - 1]
    vocab = {t['w'] for t in ts}
    contrast = arg['contrast']

    def side(w):
        if contrast == 'sh/ch':
            return 0 if w.startswith('sh') and 'ch' + w[2:] in vocab else \
                   1 if w.startswith('ch') and 'sh' + w[2:] in vocab else None
        if contrast == 'qo/o':
            return 0 if w.startswith('qo') and w[1:] in vocab else \
                   1 if w.startswith('o') and 'q' + w in vocab else None
        matches = {0 if c == 'k' else 1 for i, c in enumerate(w) if c in 'kt' and
                   w[:i] + ('t' if c == 'k' else 'k') + w[i + 1:] in vocab}
        # Mixed k/t words with matches on both sides cannot be assigned uniquely.
        return next(iter(matches)) if len(matches) == 1 else None

    groups = [[], []]
    used_pages = set()
    for t in ts:
        s = side(t['w'])
        if s is not None:
            groups[s].append(t['k'] / (t['n'] - 1))
            used_pages.add(t['page'])
    counts = {'first': len(groups[0]), 'second': len(groups[1]), 'pages': len(used_pages)}
    value = NAN if min(map(len, groups)) < MIN_TOKENS or len(used_pages) < 5 else \
        sum(groups[1]) / len(groups[1]) - sum(groups[0]) / len(groups[0])
    return (value, counts) if with_counts else value

def _stop_types(toks, k):
    """K most frequent types, including every tie at the cutoff; invariant to type renaming."""
    if not k:
        return set()
    counts = Counter(t['w'] for t in toks)
    if k > len(counts):
        return set(counts)
    threshold = sorted(counts.values(), reverse=True)[k - 1]
    return {word for word, count in counts.items() if count >= threshold}


def _directional_pair_counts(toks, arg):
    stops = _stop_types(toks, arg['drop_top'])
    directed = Counter()
    kept = 0
    for run in runs_of(toks):
        words = [t['w'] for t in run if t['w'] not in stops]
        kept += len(words)
        for i, a in enumerate(words):
            for b in words[i + 1:i + 1 + arg['window']]:
                if a != b:
                    directed[a, b] += 1
    unordered = {tuple(sorted((a, b))) for a, b in directed}
    skewed = sum(abs(directed[a, b] - directed[b, a]) > arg['excess'] for a, b in unordered)
    return {'retained_tokens': kept, 'removed_types': len(stops),
            'pair_types': len(unordered), 'skewed_types': skewed}


def _filtered_vocab_counts(toks, arg):
    stops = _stop_types(toks, arg['drop_top'])
    counts = Counter(t['w'] for t in toks if t['w'] not in stops)
    return {'retained_tokens': sum(counts.values()), 'removed_types': len(stops),
            'distinct_types': len(counts), 'singleton_types': sum(v == 1 for v in counts.values())}


# Fixed pool used by the released Parisel grille tokenizer. Longest matches first.
_BOUNDARY_UNITS = tuple(sorted({"o", "a", "q", "d", "s", "t", "k", "p", "f", "c", "ch", "sh", "ck", "cth", "cph", "ok", "ot", "ol", "or", "k2", "t2", "p2", "f2", "d2", "s2", "ch2", "sh2", "ckh", "cth2", "cfh", "dy", "dl", "dm", "ds", "dar", "dal", "iin", "in", "ir", "iir", "iiir", "al", "am", "an", "ar", "ain", "aiin", "ey", "ed", "es", "edy", "eey", "eedy", "ly", "ry", "ny", "my", "ldy"}, key=lambda x: (-len(x), x)))


def _boundary_seg(word):
    out = []
    while word:
        match = next((u for u in _BOUNDARY_UNITS if word.startswith(u)), word[0])
        out.append(match)
        word = word[len(match):]
    return out


def _boundary_eligible(t):
    w = t["w"]
    return t["unit"] == "P" and len(w) > 1 and w.isalpha() and not UNCERTAIN(w)


def _boundary_classes(toks):
    start, end = Counter(), Counter()
    for t in toks:
        if _boundary_eligible(t):
            g = _boundary_seg(t["w"])
            start[g[0]] += 1
            end[g[-1]] += 1
    if sum(start.values()) < 100:
        raise SpecError("boundary_class_rate needs at least 100 eligible training words")
    return {g: ("start" if start[g] > 2 * end[g] else "end" if end[g] > 2 * start[g] else "ambig")
            for g in set(start) | set(end)}


def _boundary_class_value(toks, classes):
    pairs = [(a, b) for run in runs_of([t for t in toks if _boundary_eligible(t)])
             for a, b in zip(run, run[1:])]
    if len(pairs) < MIN_TOKENS:
        return NAN, {"pairs": len(pairs), "unknown_pairs": 0}
    edges = [(_boundary_seg(a["w"])[-1], _boundary_seg(b["w"])[0]) for a, b in pairs]
    unknown = sum(a not in classes or b not in classes for a, b in edges)
    rate = sum(classes.get(a) == "end" and classes.get(b) == "start" for a, b in edges) / len(edges)
    return rate, {"pairs": len(edges), "unknown_pairs": unknown}


def check_statistic(st):
    if not isinstance(st, dict) or len(st) != 1:
        raise SpecError("statistic is one {name: args}")
    (name, arg), = st.items()
    if name in ("rate", "ratio"):
        _keys(arg, {"feature", "in"} | ({"of", "to"} if name == "ratio" else set()), name,
              ("feature", "in") + (("of", "to") if name == "ratio" else ()))
        check_feature(arg["feature"])
        if not isinstance(arg["in"], list) or not arg["in"] or not all(isinstance(v, (str, int)) for v in arg["in"]):
            raise SpecError(f"{name}.in is a non-empty list of values")
        if name == "ratio":
            check_select(arg["of"], "ratio.of")
            check_select(arg["to"], "ratio.to")
    elif name in ("entropy", "cond_entropy", "adjacent_mi"):
        check_feature(arg)
    elif name == "glyph_cond_entropy":
        _int(arg, 1, 3, "glyph_cond_entropy order")
    elif name == "repeat_ngram":
        _int(arg, 1, 6, "repeat_ngram n")
    elif name == "zipf_slope":
        _int(arg, 3, 100_000, "zipf_slope ranks")
    elif name in ("edge_mi", "edge_mi_break", "edge_length_cmi", "hapax_share", "mean_length", "line_repeat_excess", "repeated_pair_types"):
        if arg is not True:
            raise SpecError(f"{name} takes true")
    elif name == "line_near_excess":
        _keys(arg, {"min_length"}, name)
        _int(arg.get("min_length", 4), 1, 40, "line_near_excess min_length")
    elif name == "near_copy":
        _keys(arg, {"window", "min_length"}, name)
        _int(arg.get("window", 10), 1, 50, "near_copy window")
        _int(arg.get("min_length", 6), 1, 40, "near_copy min_length")
    elif name == "page_similarity":
        _keys(arg, {"within", "min_words"}, name, ("within",))
        _int(arg["within"], 1, 50, "page_similarity within")
        _int(arg.get("min_words", 40), 10, 10_000, "page_similarity min_words")
    elif name == "page_group_similarity":
        _keys(arg, {"by", "min_words"}, name, ("by",))
        if arg["by"] not in ("section", "language", "hand"):
            raise SpecError("page_group_similarity.by is section | language | hand")
        _int(arg.get("min_words", 40), 10, 10_000, "page_group_similarity min_words")
    elif name == "page_reuse":
        _keys(arg, {"min_length"}, name)
        _int(arg.get("min_length", 1), 1, 40, "page_reuse min_length")
    elif name == "boundary_class_rate":
        if arg is not True:
            raise SpecError("boundary_class_rate must be true")
    elif name == "matched_rightwardness":
        _keys(arg, {"contrast", "min_line", "max_line"}, name, ("contrast", "min_line", "max_line"))
        if arg["contrast"] not in ("sh/ch", "qo/o", "k/t"):
            raise SpecError("matched_rightwardness contrast is sh/ch | qo/o | k/t")
        lo = _int(arg["min_line"], 3, 200, "min_line")
        hi = _int(arg["max_line"], 3, 200, "max_line")
        if lo > hi:
            raise SpecError("min_line must not exceed max_line")
    elif name == "directional_pairs":
        _keys(arg, {"window", "drop_top", "excess"}, name, ("window", "drop_top", "excess"))
        _int(arg['window'], 1, 10, 'directional_pairs window')
        _int(arg['drop_top'], 0, 20, 'directional_pairs drop_top')
        _int(arg['excess'], 0, 1000, 'directional_pairs excess')
    elif name == "filtered_vocab":
        _keys(arg, {"measure", "drop_top"}, name, ("measure", "drop_top"))
        if arg['measure'] not in ('type_token', 'singleton_token'):
            raise SpecError('filtered_vocab.measure is type_token | singleton_token')
        _int(arg['drop_top'], 0, 20, 'filtered_vocab drop_top')
    elif name == "picture_label":
        _keys(arg, {"test", "min_objects"}, name, ("test",))
        if arg["test"] not in PICTURE_TESTS:
            raise SpecError(f"picture_label.test is one of {sorted(PICTURE_TESTS)}")
        _int(arg.get("min_objects", 20), 10, 10_000, "picture_label min_objects")
    elif name == "context_mi":
        _keys(arg, {"feature", "by"}, name, ("feature", "by"))
        check_feature(arg["feature"])
        if arg["by"] not in ("page", "section", "unit", "line_position"):
            raise SpecError("context_mi.by is page | section | unit | line_position")
    elif name == "paragraph_gap_corr":
        _keys(arg, {"symbol"}, name, ("symbol",))
        if not isinstance(arg["symbol"], str) or len(arg["symbol"]) != 1 or arg["symbol"] not in "acdefghiklmnopqrsty":
            raise SpecError("paragraph_gap_corr.symbol must be one eligible EVA character")
    else:
        raise SpecError(f"unknown statistic {name!r}")
    return name


def statistic(st, toks, all_toks):
    """The value, or NaN where the statistic is undefined on this selection (too little to count)."""
    name = check_statistic(st)
    arg = st[name]
    if name == "ratio":
        vals = set(arg["in"])
        num, den = select(all_toks, arg["of"]), select(all_toks, arg["to"])
        if len(num) < MIN_TOKENS or len(den) < MIN_TOKENS:
            return NAN
        r = lambda ts: sum(any(v in vals for v in feature(arg["feature"], t["w"])) for t in ts) / len(ts)
        a, b = r(num), r(den)
        return a / b if b else (float("inf") if a else NAN)
    if name == "picture_label":
        return _picture_label(arg["test"], all_toks, arg.get("min_objects", 20))
    if name == "matched_rightwardness":
        return _matched_rightwardness(toks, arg)
    if len(toks) < MIN_TOKENS:
        return NAN
    if name == "repeated_pair_types":
        eligible = [t for t in toks if t['unit'] == 'P' and t['w'] and not UNCERTAIN(t['w'])]
        pairs = Counter((a['w'], b['w']) for run in runs_of(eligible) for a, b in zip(run, run[1:]))
        if sum(pairs.values()) < MIN_TOKENS:
            return NAN
        return sum(n >= 2 for n in pairs.values()) / len(pairs)
    if name == "paragraph_gap_corr":
        return _paragraph_gap_value(all_toks, arg["symbol"])[0]
    if name == "directional_pairs":
        c = _directional_pair_counts(toks, arg)
        return c['skewed_types'] / c['pair_types'] if min(c['retained_tokens'], c['pair_types']) >= MIN_TOKENS else NAN
    if name == "filtered_vocab":
        c = _filtered_vocab_counts(toks, arg)
        numerator = c['distinct_types'] if arg['measure'] == 'type_token' else c['singleton_types']
        return numerator / c['retained_tokens'] if c['retained_tokens'] >= MIN_TOKENS else NAN
    if name == "rate":
        vals = set(arg["in"])
        return sum(any(v in vals for v in feature(arg["feature"], t["w"])) for t in toks) / len(toks)
    if name == "entropy":
        return _entropy(Counter(v for t in toks for v in feature(arg, t["w"])))
    if name == "edge_length_cmi":
        pairs = _edge_length_pairs(toks)
        if len(pairs) < MIN_TOKENS:
            return NAN
        joint, left, right, strata = Counter(), Counter(), Counter(), Counter()
        for a, b in pairs:
            x, y = SEG(a["w"]), SEG(b["w"])
            z = (len(x), len(y))
            joint[z, x[-1], y[0]] += 1
            left[z, x[-1]] += 1
            right[z, y[0]] += 1
            strata[z] += 1
        return sum(n * math.log2(n * strata[z] / (left[z, x] * right[z, y]))
                   for (z, x, y), n in joint.items()) / len(pairs)
    if name in ("cond_entropy", "adjacent_mi", "edge_mi"):
        runs = runs_of(toks)
        if sum(len(r) - 1 for r in runs) < MIN_TOKENS:
            return NAN
        if name == "edge_mi":
            return _mi([(SEG(x["w"])[-1], SEG(y["w"])[0]) for r in runs for x, y in zip(r, r[1:])])
        fw = lambda w: tuple(feature(arg, w))
        pairs = [(fw(x["w"]), fw(y["w"])) for r in runs for x, y in zip(r, r[1:])]
        if name == "cond_entropy":
            return _entropy(Counter(pairs)) - _entropy(Counter(p[0] for p in pairs))
        rng = random.Random(1)
        spairs = []
        for r in runs:
            ws = [t["w"] for t in r]
            rng.shuffle(ws)
            spairs += [(fw(x), fw(y)) for x, y in zip(ws, ws[1:])]
        return _mi(pairs) - _mi(spairs)
    if name == "edge_mi_break":
        return _edge_break(toks)
    if name == "glyph_cond_entropy":
        k = int(arg)
        ctx, full = Counter(), Counter()
        for r in runs_of(toks):
            s = ["_"] * k
            for t in r:
                s += list(SEG(t["w"])) + ["_"]
            for i in range(k, len(s)):
                ctx[tuple(s[i - k:i])] += 1
                full[tuple(s[i - k:i + 1])] += 1
        return _entropy(full) - _entropy(ctx)
    if name == "repeat_ngram":
        n = int(arg)
        c = Counter(tuple(t["w"] for t in r[i:i + n]) for r in runs_of(toks) for i in range(len(r) - n + 1))
        tot = sum(c.values())
        return sum(v for v in c.values() if v >= 2) / tot if tot >= MIN_TOKENS else NAN
    if name == "page_similarity":
        return _page_similarity(toks, int(arg["within"]), int(arg.get("min_words", 40)))
    if name == "page_group_similarity":
        return _page_group_similarity(toks, arg["by"], int(arg.get("min_words", 40)))
    if name in ("line_repeat_excess", "line_near_excess"):
        # Use the same current words for both lags. Require five consecutive
        # selected lines on one page so gaps do not become spurious neighbors.
        words = {}
        for t in toks:
            words.setdefault(t["line"], []).append(t["w"])
        vocabulary = {line: set(ws) for line, ws in words.items()}
        if name == "line_repeat_excess":
            L, on = 2, lambda word, voc: word in voc
        else:
            L, on = int(arg.get("min_length", 4)), lambda word, voc: any(_edit1(word, u) for u in voc)
        hits = eligible = 0
        for (page, line), ws in words.items():
            if not all((page, line - lag) in words for lag in range(1, 5)):
                continue
            near, far = vocabulary[page, line - 1], vocabulary[page, line - 4]
            for word in ws:
                if len(word) >= L:
                    eligible += 1
                    hits += on(word, near) - on(word, far)
        return hits / eligible if eligible >= MIN_TOKENS else NAN
    if name == "hapax_share":
        c = Counter(t["w"] for t in toks)
        return sum(1 for v in c.values() if v == 1) / len(c)
    if name == "zipf_slope":
        fr = [f for _w, f in Counter(t["w"] for t in toks).most_common(int(arg))]
        if len(fr) < 3:
            return NAN
        xs = [math.log10(i + 1) for i in range(len(fr))]
        ys = [math.log10(f) for f in fr]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    if name == "mean_length":
        return sum(len(SEG(t["w"])) for t in toks) / len(toks)
    if name in ("near_copy", "page_reuse"):
        by_page = {}
        for t in toks:
            by_page.setdefault(t["page"], []).append(t["w"])
        hits = n = 0
        if name == "page_reuse":
            L = int(arg.get("min_length", 1))
            for ws in by_page.values():
                seen = set()
                for w in ws:
                    if len(w) >= L:
                        n += 1
                        hits += w in seen
                    seen.add(w)
        else:
            k, L = int(arg.get("window", 10)), int(arg.get("min_length", 6))
            for ws in by_page.values():
                for i, w in enumerate(ws):
                    if len(w) < L:
                        continue
                    n += 1
                    hits += any(_edit1(w, u) for u in ws[max(0, i - k):i])
        return hits / n if n >= MIN_TOKENS else NAN
    if name == "context_mi":
        return _mi([(v, _context(t, arg["by"])) for t in toks for v in feature(arg["feature"], t["w"])])
    raise SpecError(f"unknown statistic {name!r}")


# ------------------------------------------------------------- pictures

PICTURE_TESTS = {"same_object_label", "feature_label_share", "label_in_page_text"}
MIN_PICTURE_PAIRS = 5
_FEATURE_SKIP = ("label", "order", "angle", "clock", "position", "note", "count", "ring", "group")   # layout, not looks


def _selected_pictures(records):
    """Select the best-agreeing annotation per folio, using explicit label coverage to break ties."""
    from .pictures import explicit_label

    best = {}
    for i in records:
        if i.get("verdict") != "AGREED":
            continue
        d = i.get("detail") or {}
        f = str(d.get("folio"))
        objects = d.get("objects") or []
        labels = {label for o in objects if (label := explicit_label(o))}
        rank = (i.get("agreement", 0), len(labels))
        if f not in best or rank > best[f][0]:
            best[f] = (rank, i)
    return {f: record for f, (_, record) in best.items()}


def pictures():
    from . import registry
    return {f: r["detail"]["objects"] for f, r in _selected_pictures(registry.read("picture")).items()}


def _looks(o):
    """The object's visible features, normalised, without layout fields."""
    fs = o.get("features") if isinstance(o.get("features"), dict) else {}
    return {k: str(v).strip().lower() for k, v in fs.items()
            if not any(x in k.lower() for x in _FEATURE_SKIP) and str(v).strip()}


LABEL_POSITIONS = None   # {folio: [[(locus, degrees), ...] per ring, outermost first]}; None reads ZL3b (tests set it)
_POSITION_CACHE = None
JOIN_TOLERANCE = 20.0


def label_positions():
    """Clock positions of label lines on circular pages, from the comments in the pinned ZL3b file."""
    global _POSITION_CACHE
    if LABEL_POSITIONS is not None:
        return LABEL_POSITIONS
    if _POSITION_CACHE is not None:
        return _POSITION_CACHE
    import re, os
    from . import path
    out, fn = {}, path("data", "raw", "ZL3b-n.txt")
    if os.path.exists(fn):
        line_re = re.compile(r"^<(f\d+[rv]\d*)\.(\d+),.([A-Z])(.)>\s*(?:<!(\d\d):(\d\d)>)?")
        state = {}
        for raw in open(fn, encoding="utf-8", errors="replace"):
            m = line_re.match(raw)
            if not m:
                continue
            folio, n, unit = m.group(1), m.group(2), m.group(3)
            st = state.setdefault(folio, {"seen_circle": False, "runs": [], "cur": None})
            if unit == "C":
                st["seen_circle"] = True
                st["cur"] = None
            elif unit == "L" and st["seen_circle"] and m.group(5):
                if st["cur"] is None:
                    st["cur"] = []
                    st["runs"].append(st["cur"])
                hh, mm = int(m.group(5)) % 12, int(m.group(6))
                st["cur"].append((f"{folio}.{n}", (hh * 60 + mm) / 720 * 360))
        out = {f: st["runs"] for f, st in state.items() if st["runs"]}
    _POSITION_CACHE = out
    return out


def _joined_labels(folio, objs):
    """Project each position independently of its explicit label, rejecting tied minima on either end."""
    from .pictures import _ring, circular_distance, clock_angle, unique_nearest
    runs = label_positions().get(folio) or []
    names = {2: ["outer", "inner"], 3: ["outer", "middle", "inner"]}.get(len(runs), ["outer"] if len(runs) == 1 else [])
    rings = dict(zip(names, runs))
    positions = {}
    for i, o in enumerate(objs):
        if not isinstance(o, dict):
            continue
        ring, ang = _ring(o), clock_angle(o)
        if ring not in rings:
            continue
        positions.setdefault(ring, []).append((i, ang))
    result = {}
    for ring, objects in positions.items():
        if any(ang is None for _, ang in objects):
            continue
        lines = rings[ring]
        nearest_line = {i: unique_nearest(((circular_distance(ang, deg), lab) for lab, deg in lines), JOIN_TOLERANCE)
                        for i, ang in objects}
        for lab, deg in lines:
            i = unique_nearest(((circular_distance(ang, deg), i) for i, ang in objects), JOIN_TOLERANCE)
            if i is not None and nearest_line[i] == lab:
                result[i] = lab
    return result


def _confirmed_labels(folio, selected, peers):
    from . import pictures as pic, registry
    detail = selected["detail"]
    objs = detail["objects"]
    selected_projections = _joined_labels(folio, objs)
    claims = [set() for _ in objs]
    confirmed = set()
    unmatched_conflicts = set()
    for i, obj in enumerate(objs):
        if lab := pic.explicit_label(obj):
            claims[i].add(lab)
    selected_seat = registry.picture_seat(selected)
    if not selected_seat:
        return {}
    for peer in peers:
        seat = registry.picture_seat(peer)
        if not seat or seat == selected_seat:
            continue
        other = peer["detail"]
        score = registry.picture_agreement(detail, other)
        if score is None or score < registry.PICTURE_AGREED:
            continue
        peer_projections = _joined_labels(folio, other["objects"])
        pairs = pic.compare_labels(detail, other, selected_projections, peer_projections)
        unmatched_conflicts.update(pic.conflicting_unpaired_loci(detail, other, pairs))
        for pair in pairs:
            if lab := pic.explicit_label(other["objects"][pair.right]):
                claims[pair.left].add(lab)
            if pair.locus:
                confirmed.add(pair.left)
                claims[pair.left].add(pair.locus)
    owners = Counter(lab for values in claims for lab in values)
    return {i: lab for i, values in enumerate(claims) if i in confirmed and len(values) == 1
            for lab in values if owners[lab] == 1 and (i, lab) not in unmatched_conflicts}


@lru_cache(maxsize=1)
def _confirmed_pictures():
    from . import registry
    records = registry.read("picture")
    out = {}
    for folio, selected in _selected_pictures(records).items():
        peers = (r for r in records if r.get("verdict") == "AGREED" and r["detail"]["folio"] == folio)
        labels = _confirmed_labels(folio, selected, peers)
        out[folio] = [(selected["detail"]["objects"][i], lab) for i, lab in labels.items()]
    return out


def _picture_objects(all_toks):
    """[(folio, kind, looks, label line id, label words)] for agreed objects whose label line is on these pages."""
    lines = {}
    for t in all_toks:
        if t["unit"] == "L" and t.get("lid"):
            lines.setdefault((t["folio"], t["lid"]), []).append(t["w"])
    out = []
    for folio, objects in _confirmed_pictures().items():
        for o, lab in objects:
            if (folio, lab) in lines:
                out.append((folio, o.get("kind"), _looks(o), lab, tuple(lines[folio, lab])))
    return out


def _glyph_pairs(words):
    out = set()
    for w in words:
        g = SEG(w)
        if len(g) >= 2:
            out.add(("first", tuple(g[:2])))
            out.add(("last", tuple(g[-2:])))
    return out


def _rate_diff(pairs, is_a, hit, undefined):
    a = [hit(x, y) for x, y in pairs if is_a(x, y)]
    b = [hit(x, y) for x, y in pairs if not is_a(x, y)]
    if len(a) < MIN_PICTURE_PAIRS or len(b) < MIN_PICTURE_PAIRS:
        return undefined(f"{len(a)} alike and {len(b)} other pairs; need at least {MIN_PICTURE_PAIRS} in each group")
    return sum(a) / len(a) - sum(b) / len(b)


def _picture_label(test, all_toks, min_objects, *, required=False):
    objs = _picture_objects(all_toks)

    def undefined(reason):
        if required:
            raise SpecError(f"picture_label {test} is undefined: {len(objs)} independently confirmed object-label "
                            f"links on the scored pages; {reason}")
        return NAN

    if len(objs) < min_objects:
        if not required:
            return NAN
        confirmed = _confirmed_pictures()
        agreed = confirmed.keys() & {t["folio"] for t in all_toks}
        available = sum(len(confirmed[folio]) for folio in agreed)
        if not agreed:
            reason = "none of the scored folios has an AGREED picture annotation"
        elif not available:
            reason = ("visual agreement exists, but no drawing has an unambiguous label line supported by "
                      "two independent seats without a conflicting assignment")
        elif available != len(objs):
            reason = f"only {len(objs)} of {available} confirmed loci have label words on these pages"
        else:
            reason = "more drawings with independently confirmed label lines are needed"
        return undefined(f"need at least {min_objects} links. {reason}")
    if test == "label_in_page_text":
        text = {}
        for t in all_toks:
            if t["unit"] == "P":
                text.setdefault(t["folio"], set()).add(t["w"])
        if len(text) < 2:
            return undefined(f"{len(text)} folios have paragraph text; need at least 2")
        own, other, n = 0.0, 0.0, 0
        for folio, _, _, _, ws in objs:
            long_ = {w for w in ws if len(SEG(w)) >= 3}
            if folio not in text or not long_:
                continue
            n += 1
            own += bool(long_ & text[folio])
            rest = [f for f in text if f != folio]
            other += sum(bool(long_ & text[f]) for f in rest) / len(rest)
        if n < min_objects:
            return undefined(f"links with a word of at least 3 glyphs and paragraph text on their folio: {n}; "
                             f"need at least {min_objects}")
        return (own - other) / n
    pairs = [(x, y) for i, x in enumerate(objs) for y in objs[i + 1:]]
    if test == "feature_label_share":
        def alike(x, y):
            return x[1] == y[1] and any(x[2].get(k) == v for k, v in y[2].items())
        return _rate_diff(pairs, alike, lambda x, y: bool(_glyph_pairs(x[4]) & _glyph_pairs(y[4])), undefined)
    # same_object_label: across folios only
    pairs = [(x, y) for x, y in pairs if x[0] != y[0]]
    def same(x, y):
        if x[1] != y[1]:
            return False
        shared = set(x[2]) & set(y[2])
        return len(shared) >= 2 and sum(x[2][k] == y[2][k] for k in shared) >= 0.8 * len(shared)
    def close(x, y):
        return any(a == b or _edit1(a, b) for a in x[4] for b in y[4])
    return _rate_diff(pairs, same, close, undefined)


def _shuffle_picture_labels(all_toks, rng):
    """Deal whole label lines among the label lines agreed objects point at; everything else stays."""
    targets = sorted({o[3] for o in _picture_objects(all_toks)})
    if len(targets) < 2:
        return all_toks
    words = {lid: [t["w"] for t in all_toks if t.get("lid") == lid] for lid in targets}
    src = list(targets)
    rng.shuffle(src)
    new = dict(zip(targets, (words[s] for s in src)))
    out, done = [], set()
    for t in all_toks:
        lid = t.get("lid")
        if lid not in new:
            out.append(t)
        elif lid not in done:
            done.add(lid)
            ws = new[lid]
            out += [dict(t, w=w, k=k, n=len(ws)) for k, w in enumerate(ws)]
    return out


# ------------------------------------------------------------- nulls and references

# statistics whose features mean the same thing in another script, so a reference language can be measured
LAYOUT_FREE = {"entropy", "hapax_share", "zipf_slope", "mean_length", "repeat_ngram", "cond_entropy", "adjacent_mi",
               "edge_mi", "near_copy", "glyph_cond_entropy", "page_reuse", "line_repeat_excess",
               "line_near_excess", "directional_pairs", "filtered_vocab"}
MIN_REFERENCES = 3

_NULL_KEY = {"shuffle_within_line": lambda t: t["line"], "shuffle_within_page": lambda t: t["page"],
             "shuffle_within_line_length": None,
             "shuffle_within_section": lambda t: t["section"], "shuffle_across_pages": lambda t: 0,
             "shuffle_within_paragraph": lambda t: t.get("para"),
             "shuffle_labels": lambda t: t["section"], "shuffle_labels_within_page": lambda t: t["page"],
             "shuffle_page_labels": None, "rotate_page_labels": None, "shuffle_picture_labels": None}
_PAGE_LABELS = ("section", "language", "hand")


def _relabelled(all_toks, order, labels):
    """The same words on the same pages; page i (in page order) now carries labels[i]."""
    new = dict(zip(order, labels))
    return [dict(t, **dict(zip(_PAGE_LABELS, new[t["page"]]))) for t in all_toks]


def _page_labels(all_toks):
    lab = {}
    for t in all_toks:
        lab[t["page"]] = tuple(t[k] for k in _PAGE_LABELS)
    order = sorted(lab)
    return order, [lab[p] for p in order]


def _edge_length_pairs(toks):
    eligible = [t for t in toks if t["unit"] == "P" and not UNCERTAIN(t["w"]) and SEG(t["w"])]
    return [(a, b) for run in runs_of(eligible) for a, b in zip(run, run[1:])]


def _line_length_groups(toks):
    groups = {}
    for i, t in enumerate(toks):
        if t["unit"] == "P" and not UNCERTAIN(t["w"]) and SEG(t["w"]):
            groups.setdefault((t["line"], len(SEG(t["w"]))), []).append(i)
    return list(groups.values())


def _edge_length_movable_pairs(toks):
    movable = {(toks[i]["line"], toks[i]["k"])
               for g in _line_length_groups(toks) if len({toks[j]["w"] for j in g}) > 1 for i in g}
    return sum((a["line"], a["k"]) in movable or (b["line"], b["k"]) in movable
               for a, b in _edge_length_pairs(toks))


def _shuffled(all_toks, how, rng, statistic_name=None):
    if how not in _NULL_KEY:
        raise SpecError(f"unknown null {how!r}")
    if how == "shuffle_picture_labels":
        return _shuffle_picture_labels(all_toks, rng)
    if how == "shuffle_page_labels":
        order, labels = _page_labels(all_toks)
        rng.shuffle(labels)
        return _relabelled(all_toks, order, labels)
    if how == "shuffle_within_line_length":
        words = [t["w"] for t in all_toks]
        for g in _line_length_groups(all_toks):
            ws = [words[i] for i in g]
            rng.shuffle(ws)
            for i, w in zip(g, ws):
                words[i] = w
        return [dict(t, w=w) for t, w in zip(all_toks, words)]
    if statistic_name == "boundary_class_rate" and how == "shuffle_within_line":
        words = [t["w"] for t in all_toks]
        idx = {(t["line"], t["k"]): i for i, t in enumerate(all_toks)}
        for run in runs_of([t for t in all_toks if _boundary_eligible(t)]):
            indices = [idx[(t["line"], t["k"])] for t in run]
            ws = [words[i] for i in indices]
            rng.shuffle(ws)
            for i, w in zip(indices, ws):
                words[i] = w
        return [dict(t, w=w) for t, w in zip(all_toks, words)]
    if statistic_name == "repeated_pair_types" and how == "shuffle_within_line":
        eligible = [t for t in all_toks if t['unit'] == 'P' and t['w'] and not UNCERTAIN(t['w'])]
        indexes = {(t['line'], t['k']): i for i, t in enumerate(all_toks)}
        words = [t['w'] for t in all_toks]
        for run in runs_of(eligible):
            idx = [indexes[t['line'], t['k']] for t in run]
            shuffled = [words[i] for i in idx]
            rng.shuffle(shuffled)
            for i, word in zip(idx, shuffled):
                words[i] = word
        return [dict(t, w=w) for t, w in zip(all_toks, words)]
    if statistic_name == "paragraph_gap_corr" and how in ("shuffle_within_paragraph", "shuffle_within_line"):
        groups = (_paragraph_gap_groups(all_toks) if how == "shuffle_within_paragraph"
                  else _paragraph_gap_groups(all_toks, by_line=True))
        words = [t["w"] for t in all_toks]
        for group in groups:
            shuffled = [words[i] for i in group]
            rng.shuffle(shuffled)
            for i, word in zip(group, shuffled):
                words[i] = word
        return [dict(t, w=w) for t, w in zip(all_toks, words)]
    key = _NULL_KEY[how]
    groups = {}
    for i, t in enumerate(all_toks):
        groups.setdefault(key(t), []).append(i)
    words = [t["w"] for t in all_toks]
    for g in groups.values():
        if how.startswith("shuffle_labels"):
            # reassign which words of the stratum are the labels; every other word stays in place
            lab = [i for i in g if all_toks[i]["unit"] == "L"]
            if not lab or len(lab) == len(g):
                continue
            chosen = rng.sample(g, len(lab))
            cs = set(chosen)
            new_labels = [words[i] for i in chosen]
            vacated = [i for i in chosen if all_toks[i]["unit"] != "L"]
            displaced = [words[i] for i in lab if i not in cs]
            rng.shuffle(displaced)
            for i, w in zip(lab, new_labels):
                words[i] = w
            for i, w in zip(vacated, displaced):
                words[i] = w
        else:
            ws = [words[i] for i in g]
            rng.shuffle(ws)
            for i, w in zip(g, ws):
                words[i] = w
    return [dict(t, w=w) for t, w in zip(all_toks, words)]


_GAP_EVA = frozenset("acdefghiklmnopqrsty")
_GAP_LAGS = tuple(range(1, 6))
_GAP_MIN_PAIRS = 25


def _paragraph_gap_groups(all_toks, by_line=False):
    """Return word indices in complete, certain, alphabet-eligible paragraph runs."""
    lines = {}
    for i, token in enumerate(all_toks):
        if token["unit"] == "P":
            lines.setdefault(token["line"], []).append((i, token))
    ordered = sorted(lines.values(), key=lambda row: (row[0][1]["page"], row[0][1]["line"][1]))
    groups, run = [], []
    previous_page = previous_para = current_page = None
    paragraph_index = -1

    def flush():
        nonlocal run
        if run:
            groups.append(run)
            run = []

    for row in ordered:
        indices, tokens_on_line = [v[0] for v in row], [v[1] for v in row]
        first = tokens_on_line[0]
        if first["page"] != current_page:
            current_page, paragraph_index = first["page"], -1
        if first["pfl"]:
            paragraph_index += 1
        paragraph = (first["page"], paragraph_index)
        complete = (len(tokens_on_line) == first["n"]
                    and [t["k"] for t in tokens_on_line] == list(range(first["n"]))
                    and all(not ivtff.uncertain(t["w"]) and t["w"]
                            and set(t["w"]) <= _GAP_EVA for t in tokens_on_line))
        contiguous = first["page"] == previous_page and paragraph == previous_para
        if not complete or (run and not contiguous):
            flush()
        if complete:
            if by_line:
                groups.append(indices)
            else:
                run.extend(indices)
        previous_page = first["page"] if complete else None
        previous_para = paragraph if complete else None
        if by_line:
            flush()
    flush()
    return groups


def _gap_curve(runs, symbol):
    """Pooled endpoint-centered gap correlations at the fixed lags 1..5."""
    centered = []
    for words in runs:
        hits, offset = [], 0
        for word in words:
            hits.extend(offset + i for i, char in enumerate(word) if char == symbol)
            offset += len(word) + 1
        gaps = [b - a for a, b in zip(hits, hits[1:])]
        if len(gaps) > 2:
            center = mean(gaps)
            centered.append([gap - center for gap in gaps])
    curve = {}
    for lag in _GAP_LAGS:
        left, right = [], []
        for gaps in centered:
            if len(gaps) > lag + 1:
                left.extend(gaps[:-lag])
                right.extend(gaps[lag:])
        pairs = len(left)
        xx, yy = fsum(x*x for x in left), fsum(y*y for y in right)
        xy = fsum(x*y for x, y in zip(left, right))
        value = xy / sqrt(xx * yy) if pairs >= _GAP_MIN_PAIRS and xx and yy else None
        curve[lag] = (value, pairs)
    return curve


def _paragraph_gap_value(all_toks, symbol):
    index_runs = _paragraph_gap_groups(all_toks)
    runs = [[all_toks[i]["w"] for i in indices] for indices in index_runs]
    curve = _gap_curve(runs, symbol)
    if any(curve[lag][0] is None for lag in _GAP_LAGS):
        return NAN, {lag: curve[lag][1] for lag in _GAP_LAGS}
    return mean(curve[lag][0] for lag in _GAP_LAGS), {lag: curve[lag][1] for lag in _GAP_LAGS}


def _reference_values(st, toks):
    """The statistic on each reference language poured into the layout of the selected tokens."""
    from .facts import ref_langs, ref_words
    vals = {}
    for lang in ref_langs():
        ws = ref_words(lang, len(toks))
        if len(ws) < len(toks):
            continue  # too short to match the size: a smaller sample would bias every plug-in statistic
        ts = [dict(t, w=w) for t, w in zip(toks, ws)]
        vals[lang] = statistic(st, ts, ts)
    if len(vals) < MIN_REFERENCES:
        raise SpecError(f"fewer than {MIN_REFERENCES} reference languages have {len(toks)} words to match this selection")
    return vals


def check(spec):
    """Shape and bounds of a spec; raises SpecError. Nothing here touches the pages."""
    _keys(spec, {"scope", "select", "statistic", "compare"}, "spec", ("statistic", "compare"))
    scope = spec.get("scope") or {}
    _keys(scope, {"hand", "language", "section"}, "scope")
    if not all(isinstance(v, (str, int)) and not isinstance(v, bool) for v in scope.values()):
        raise SpecError("scope values are strings")
    check_select(spec.get("select"))
    name = check_statistic(spec["statistic"])
    if name == "repeated_pair_types":
        sel = spec.get("select") or {}
        if (set(sel.get("unit") or []) != {"P"} or sel.get("position", "any") != "any"
                or not sel.get("certain", True) or "language" in sel):
            raise SpecError("repeated_pair_types requires all certain paragraph words")
    if name in ("edge_length_cmi", "boundary_class_rate"):
        sel = spec.get("select") or {}
        if (set(sel.get("unit") or []) != {"P"} or sel.get("position", "any") != "any"
                or not sel.get("certain", True) or (name == "boundary_class_rate" and "language" in sel)):
            raise SpecError(f"{name} requires certain P words at position any")
    if name == "paragraph_gap_corr":
        selection = spec.get("select") or {}
        if (set(selection.get("unit") or []) != {"P"} or selection.get("position", "any") != "any"
                or not selection.get("certain", True) or "language" in selection):
            raise SpecError("paragraph_gap_corr requires the complete certain paragraph-text selection")
    cmp_ = spec["compare"]
    if not isinstance(cmp_, list) or not 1 <= len(cmp_) <= MAX_COMPARE:
        raise SpecError(f"compare is a list of 1 to {MAX_COMPARE} tests (a claim with no test is not a claim)")
    for c in cmp_:
        if not isinstance(c, dict) or sum(k in c for k in ("op", "reference", "null", "page_bootstrap")) != 1:
            raise SpecError(f"unknown comparison {c!r}")
        if "page_bootstrap" in c:
            _keys(c, {"page_bootstrap", "value"}, "comparison", ("value",))
            if name != "matched_rightwardness" or c["page_bootstrap"] not in ("lower", "upper"):
                raise SpecError("page_bootstrap is lower | upper and only for matched_rightwardness")
            _num(c["value"], "bootstrap threshold")
        elif "op" in c:
            _keys(c, {"op", "value"}, "comparison", ("value",))
            if c["op"] not in (">=", "<=", ">", "<"):
                raise SpecError(f"unknown op {c['op']!r}")
            _num(c["value"], "threshold value")
        elif "reference" in c:
            _keys(c, {"reference", "margin"}, "comparison")
            if c["reference"] not in ("below_all", "above_all", "inside_range"):
                raise SpecError(f"unknown reference test {c['reference']!r}")
            if _num(c.get("margin", 0), "margin") < 0:
                raise SpecError("margin must be >= 0 (a negative margin passes anything)")
            if name not in LAYOUT_FREE:
                raise SpecError(f"reference comparison needs a statistic that carries over to another script ({sorted(LAYOUT_FREE)})")
        else:
            _keys(c, {"null", "perms", "p", "direction"}, "comparison")
            if c["null"] not in _NULL_KEY:
                raise SpecError(f"unknown null {c['null']!r}")
            if (name == "edge_length_cmi") != (c["null"] == "shuffle_within_line_length"):
                raise SpecError("edge_length_cmi uses only shuffle_within_line_length; this null is reserved for it")
            if name == "boundary_class_rate" and c["null"] != "shuffle_within_line":
                raise SpecError("boundary_class_rate uses only shuffle_within_line")
            if name == "paragraph_gap_corr" and c["null"] not in ("shuffle_within_paragraph", "shuffle_within_line"):
                raise SpecError("paragraph_gap_corr uses shuffle_within_paragraph or shuffle_within_line")
            if name == "repeated_pair_types" and c["null"] != "shuffle_within_line":
                raise SpecError("repeated_pair_types uses only shuffle_within_line")
            if c["null"] == "shuffle_within_paragraph" and name != "paragraph_gap_corr":
                raise SpecError("shuffle_within_paragraph is defined only for paragraph_gap_corr")
            if c.get("direction", "greater") not in ("greater", "less"):
                raise SpecError("direction is greater or less")
            alpha = _num(c.get("p", 0.01), "p")
            if not 0 < alpha <= MAX_ALPHA:
                raise SpecError(f"p must be in (0, {MAX_ALPHA}]")
            perms = min(_int(c.get("perms", 200), 1, 10 ** 9, "perms"), MAX_PERMS)
            if 1 / (perms + 1) > alpha:
                raise SpecError(f"{perms} permutations cannot reach p <= {alpha}")
    return spec


def canonical(spec):
    """What a spec measures and how it is tested, without the numbers a resubmission could turn:
    thresholds, margins, alpha and the permutation count. Two specs with the same canonical form are
    the same claim for the record (vs/registry.py), however their JSON is ordered or defaulted."""
    check(spec)
    def sel(selection):
        selection = selection or {}
        result = {"unit": sorted(set(selection.get("unit") or [])), "position": selection.get("position", "any"),
                  "certain": selection.get("certain", True)}
        if "language" in selection:
            result["language"] = selection["language"]
        return result

    (name, arg), = spec["statistic"].items()
    if name in ("rate", "ratio"):
        arg = dict(arg, **{"in": sorted(set(arg["in"]), key=str)})
        if name == "ratio":
            arg = dict(arg, of=sel(arg["of"]), to=sel(arg["to"]))
    elif name == "near_copy":
        arg = {"window": arg.get("window", 10), "min_length": arg.get("min_length", 6)}
    elif name == "page_reuse":
        arg = {"min_length": arg.get("min_length", 1)}
    elif name == "line_near_excess":
        arg = {"min_length": arg.get("min_length", 4)}
    elif name == "directional_pairs":
        # Pair-level excess is a tunable threshold, not a new measurement.
        arg = {"window": arg["window"], "drop_top": arg["drop_top"]}
    elif name == "page_group_similarity":
        arg = {"by": arg["by"], "min_words": arg.get("min_words", 40)}
    elif name == "picture_label":
        arg = {"test": arg["test"], "min_objects": arg.get("min_objects", 20)}
    tests = set()
    for c in spec["compare"]:
        if "page_bootstrap" in c:
            tests.add("page_bootstrap:" + c["page_bootstrap"])
        elif "op" in c:
            tests.add("op" + c["op"][0])
        elif "reference" in c:
            tests.add("reference:" + c["reference"])
        else:
            tests.add(f"null:{c['null']}:{c.get('direction', 'greater')}")
    return {"scope": {k: str(v) for k, v in sorted((spec.get("scope") or {}).items()) if v},
            "select": sel(spec.get("select")), "statistic": {name: arg}, "tests": sorted(tests)}


def evaluate(spec, pages, train_pages=None):
    """Returns {"value", "pass", "detail"} for a spec on these pages."""
    check(spec)
    all_toks = tokens(pages, spec.get("scope"))
    toks = select(all_toks, spec.get("select"))
    if not toks:
        raise SpecError("the selection is empty")
    st = spec["statistic"]
    (name, _arg), = st.items()
    boundary_classes = None
    if name == "boundary_class_rate":
        if train_pages is None:
            raise SpecError("boundary_class_rate requires separate training pages")
        if {p.id for p in pages} & {p.id for p in train_pages}:
            raise SpecError("boundary_class_rate training and evaluation pages overlap")
        boundary_classes = _boundary_classes(tokens(train_pages, spec.get("scope")))
        v, boundary_detail = _boundary_class_value(toks, boundary_classes)
    elif name == "picture_label":
        v = _picture_label(_arg["test"], all_toks, _arg.get("min_objects", 20), required=True)
    else:
        v = statistic(st, toks, all_toks)
    if v != v:
        if name == "paragraph_gap_corr":
            _unused, support = _paragraph_gap_value(all_toks, st[name]["symbol"])
            raise SpecError(f"paragraph_gap_corr needs at least {_GAP_MIN_PAIRS} supported pairs at each lag 1..5; "
                            f"observed pairs by lag: {support}")
        raise SpecError(f"the statistic is undefined on this selection ({len(toks)} tokens): it needs at least "
                        f"{MIN_TOKENS} tokens, and {MIN_TOKENS} pairs / n-grams / long words where it counts those")
    finite = math.isfinite(v)
    detail, ok = {"tokens": len(toks)}, finite
    if name == "boundary_class_rate":
        detail["boundary_class_rate"] = boundary_detail
    if 'matched_rightwardness' in st:
        _, detail['matched_rightwardness'] = _matched_rightwardness(toks, st['matched_rightwardness'], True)
    if "edge_length_cmi" in st:
        movable = _edge_length_movable_pairs(toks)
        detail.update(adjacent_pairs=len(_edge_length_pairs(toks)), potentially_movable_pairs=movable)
        if any(c.get("null") == "shuffle_within_line_length" for c in spec["compare"]) and movable < MIN_TOKENS:
            raise SpecError("shuffle_within_line_length needs at least 30 potentially movable adjacent pairs")
    if name == 'directional_pairs':
        detail['directional_pairs'] = _directional_pair_counts(toks, st[name])
    elif name == 'filtered_vocab':
        detail['filtered_vocab'] = _filtered_vocab_counts(toks, st[name])
    if "paragraph_gap_corr" in st:
        _value, support = _paragraph_gap_value(all_toks, st["paragraph_gap_corr"]["symbol"])
        detail["pairs_by_lag"] = support
    if not finite:
        detail["not_finite"] = "a value that is not finite never passes"
    for c in spec["compare"]:
        if "page_bootstrap" in c:
            # Resample pages as blocks, including repeated pages with distinct identifiers.
            if detail['matched_rightwardness']['pages'] < 10:
                raise SpecError("page bootstrap needs at least 10 pages with matched tokens")
            by_page = {}
            for t in toks:
                by_page.setdefault(t['page'], []).append(t)
            ids = list(by_page)
            rng = random.Random(408)
            values = []
            for _ in range(399):
                sampled = [dict(t, page=i) for i, page in enumerate(rng.choices(ids, k=len(ids)))
                           for t in by_page[page]]
                sv = statistic(st, sampled, sampled)
                if not math.isfinite(sv):
                    raise SpecError("page bootstrap produced an undefined replicate; sample is too sparse")
                values.append(sv)
            values.sort()
            lo, hi = values[9], values[389]
            direction = c['page_bootstrap']
            r = lo > float(c['value']) if direction == 'lower' else hi < float(c['value'])
            detail['page_bootstrap:' + direction] = {'lower': round(lo, 6), 'upper': round(hi, 6),
                                                     'draws': 399, 'threshold': float(c['value']), 'pass': r}
            ok &= r
        elif "op" in c:
            x = float(c["value"])
            r = finite and {">=": v >= x, "<=": v <= x, ">": v > x, "<": v < x}[c["op"]]
            detail[f"{c['op']} {x}"] = r
            ok &= r
        elif "reference" in c:
            refs = _reference_values(st, toks)
            m = float(c.get("margin", 0))
            lo, hi = min(refs.values()), max(refs.values())
            r = all(math.isfinite(x) for x in refs.values()) and \
                {"below_all": v <= lo - m, "above_all": v >= hi + m, "inside_range": lo <= v <= hi}[c["reference"]]
            detail["reference"] = {k: round(x, 5) for k, x in refs.items()}
            ok &= r
        else:
            perms = min(int(c.get("perms", 200)), MAX_PERMS)
            rng = random.Random(408)
            more, moved = 0, False
            greater = c.get("direction", "greater") == "greater"
            if c["null"] == "rotate_page_labels":
                order, labels = _page_labels(all_toks)
                perms = len(order) - 1          # every shift of the labels along the page order, once
                if perms < 1:
                    raise SpecError("rotate_page_labels needs at least two pages")
            for k in range(perms):
                if c["null"] == "rotate_page_labels":
                    sh = _relabelled(all_toks, order, labels[k + 1:] + labels[:k + 1])
                else:
                    sh = _shuffled(all_toks, c["null"], rng, name)
                if name == "boundary_class_rate":
                    sv, _ = _boundary_class_value(select(sh, spec.get("select")), boundary_classes)
                else:
                    sv = statistic(st, select(sh, spec.get("select")), sh)
                moved |= sv == sv and abs(sv - v) > 1e-12 if finite else sv != v
                # an undefined permuted value counts against the claim
                more += (sv != sv) or ((sv >= v) if greater else (sv <= v))
            if not moved:
                raise SpecError(f"{c['null']} cannot move this statistic on this selection: it is no test of it")
            p = (more + 1) / (perms + 1)
            r = p <= float(c.get("p", 0.01))
            detail[c["null"]] = {"p": round(p, 4), "perms": perms}
            ok &= r
    return {"value": round(v, 6) if isinstance(v, float) and finite else v, "pass": bool(ok), "detail": detail}
