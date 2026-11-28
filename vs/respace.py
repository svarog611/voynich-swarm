"""Respaced plaintext: reading a decipherment whose letters run on without the plaintext's spaces.

Some ciphers do not keep words as words. In the letter-words family (vs/families/letter_words.py)
each Voynich word carries one or two letters, so the spaces between Voynich words say nothing
about where a plaintext word ends. A key file declares this with "plaintext": "respaced". The
judge then treats each page's decoded letters as one stream and asks the same four questions in
a form that fits:

 (a) COMPRESSES. The page's letters are coded by a character model of the target language with
     its spaces removed (trained on the corpus plus the key's own reading of the training pages);
     each Voynich word then pays for how many letters it carries (an adaptive count) and for
     which word it is among the words with that reading (the adaptive choice code the judge
     already uses). A complete code, so an honest upper bound.
 (b) READS. Each page's stream is cut into words of the target language (most probable cut under
     the corpus word frequencies). Per section, in chunks: bits per letter under the spaceless
     model, the share of letters that fall inside known words of three or more letters, and the
     type/token ratio of the cut, each against genuine text put through the SAME treatment
     (spaces removed, pages of the same sizes, the same cut), clean and with 2% noise.
 (3) KILL TESTS. As before, measured on streams: word order shuffled within pages must cost at
     least half of what shuffling genuine text dealt into one- and two-letter groups costs.

Nothing here reads the manuscript on its own; every threshold comes from genuine target text.
"""
import math, random
from collections import Counter
from functools import lru_cache
from . import lm, refcorpus

MAX_WORD = 24          # longest word the cut considers
MIN_KNOWN = 3          # coverage counts letters inside known words at least this long
GROUP_CAP = 4          # letters per Voynich word are counted 0, 1, 2, 3, 4+


@lru_cache(maxsize=None)
def stream_model(lang, order=5):
    """Character model of the language with its spaces removed."""
    return lm.KN([list(b.replace(" ", "")) for b in refcorpus.train_text(lang)], order, vocab=refcorpus.alphabet(lang))


@lru_cache(maxsize=None)
def _lexicon(lang, min_count=2):
    cnt = Counter(w for b in refcorpus.train_text(lang) for w in b.split())
    total = sum(cnt.values())
    known = {w: math.log2(c / total) for w, c in cnt.items() if c >= min_count and len(w) <= MAX_WORD}
    return known, -(math.log2(total) + 8.0)   # a letter outside any known word: dearer than the rarest word


def cut(letters, lang):
    """The most probable cut of a letter stream into words of the language; a letter no known
    word covers stands alone. Returns the list of words."""
    known, stray = _lexicon(lang)
    n = len(letters)
    best = [0.0] + [-float("inf")] * n
    back = [0] * (n + 1)
    for i in range(1, n + 1):
        best[i], back[i] = best[i - 1] + stray, i - 1
        for j in range(max(0, i - MAX_WORD), i):
            lp = known.get(letters[j:i])
            if lp is not None and best[j] + lp > best[i]:
                best[i], back[i] = best[j] + lp, j
    out, i = [], n
    while i > 0:
        out.append(letters[back[i]:i])
        i = back[i]
    return out[::-1]


def metrics(lang, pages):
    """pages: list of letter streams (one per page). The measures of criterion (b) on them."""
    known, _ = _lexicon(lang)
    cm = stream_model(lang)
    letters = sum(len(p) for p in pages)
    if not letters:
        return {"H": float("inf"), "cov": 0.0, "ttr": 0.0, "n": 0, "letters": 0}
    words = [w for p in pages for w in cut(p, lang)]
    covered = sum(len(w) for w in words if len(w) >= MIN_KNOWN and w in known)
    return {"H": sum(cm.bits(list(p)) for p in pages if p) / letters, "cov": covered / letters,
            "ttr": len(set(words)) / len(words), "n": len(words), "letters": letters}


def chunked(pages, letters_per_chunk):
    """Whole pages gathered into chunks of at least letters_per_chunk letters."""
    out, cur, size = [], [], 0
    for p in pages:
        if not p:
            continue
        cur.append(p)
        size += len(p)
        if size >= letters_per_chunk:
            out.append(cur)
            cur, size = [], 0
    return out


def _as_pages(letters, sizes):
    """Deal a long letter stream into pages of the given sizes, cycling through them."""
    out, i, k = [], 0, 0
    sizes = [s for s in sizes if s > 0] or [200]
    while i + sizes[k % len(sizes)] <= len(letters):
        out.append(letters[i:i + sizes[k % len(sizes)]])
        i += sizes[k % len(sizes)]
        k += 1
    return out


def _noisy(letters, lang, rate, seed):
    rng = random.Random(seed)
    alpha = refcorpus.alphabet(lang)
    return "".join(rng.choice(alpha) if rng.random() < rate else ch for ch in letters)


def _shuffled_groups(page, rng):
    """A page's letters dealt into one- and two-letter groups, the groups shuffled."""
    groups, i = [], 0
    while i < len(page):
        k = 2 if rng.random() < 0.5 else 1
        groups.append(page[i:i + k])
        i += k
    rng.shuffle(groups)
    return "".join(groups)


@lru_cache(maxsize=64)
def calibration(lang, sizes, params):
    """Bands for criterion (b) from genuine text the models never saw, put through the same
    treatment as a respaced reading: spaces removed, dealt into pages of the reading's own sizes,
    cut into words. sizes: tuple of page sizes in letters. params: the judge's PARAMS as a tuple."""
    P = dict(params)
    clean = "".join(refcorpus.calib_text(lang)).replace(" ", "")
    per_chunk = P["chunk_tokens"] * 5
    ms, gaps = [], []
    rng = random.Random(P["seed"])
    for text in (clean, _noisy(clean, lang, P["calib_noise"], 11)):
        for ch in chunked(_as_pages(text, sizes), per_chunk):
            ms.append(metrics(lang, ch))
    for ch in chunked(_as_pages(clean, sizes), per_chunk)[:60]:
        gaps.append(metrics(lang, [_shuffled_groups(p, rng) for p in ch])["H"] - metrics(lang, ch)["H"])
    if not ms:
        raise ValueError(f"too little calibration text in {lang} for pages of these sizes")
    q = _quantile
    lo, hi = P["ttr_band"]
    return {"chunks": len(ms), "H_max": q([m["H"] for m in ms], P["fluency_q"]),
            "cov_min": q([m["cov"] for m in ms], P["coverage_q"]),
            "ttr_lo": q([m["ttr"] for m in ms], lo), "ttr_hi": q([m["ttr"] for m in ms], hi),
            "H_median": q([m["H"] for m in ms], 0.5), "order_gap": sum(gaps) / len(gaps) if gaps else 0.0}


def shape_rows(lang, n, sizes, params):
    """Rows (top-3 letter share, hapax share, repeated four-word share) of genuine text cut the
    same way, in samples of n words: the respaced counterpart of judge._shape_calib."""
    from .judge import _rep4
    P = dict(params)
    clean = "".join(refcorpus.calib_text(lang)).replace(" ", "")
    words = []
    for text in (clean, _noisy(clean, lang, P["calib_noise"], 12)):
        for p in _as_pages(text, sizes):
            words.extend(cut(p, lang))
    rows = []
    for i in range(0, max(len(words) - n, 1), max(n, 1)):
        w = words[i:i + n]
        if len(w) < n:
            break
        letters = Counter(ch for x in w for ch in x)
        tot = sum(letters.values()) or 1
        cw = Counter(w)
        rows.append((sum(v for _k, v in letters.most_common(3)) / tot,
                     sum(1 for v in cw.values() if v == 1) / max(len(cw), 1), _rep4(w)))
    return rows


def _quantile(xs, q):
    xs = sorted(xs)
    i = q * (len(xs) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def page_streams(dec, pages):
    """{page id: the page's decoded letters as one stream}."""
    return {p.id: "".join(dec.raw(w).replace(" ", "") for ln in p.lines for w in ln.words) for p in pages}


def description_length(dec, base, train_pages, eval_pages, lang, params):
    """Criterion (a) for a respaced reading. Same return shape as judge.description_length."""
    from . import families
    from .judge import program_bits, compressed_bits, canonical, _literal_word_bits
    from .letter_word_preimage import FactorizedChoices
    prog_bits = program_bits(families.source_file(dec.sub["family"]))
    key_bits = compressed_bits(canonical(dec.key))
    group = lambda w: dec.raw(w).replace(" ", "")
    tr_streams = [s for s in page_streams(dec, train_pages).values() if s]
    alphabet = refcorpus.alphabet(lang)
    cm = lm.KN([list(b.replace(" ", "")) for b in refcorpus.train_text(lang)] + [list(s) for s in tr_streams],
               params["char_order"], vocab=alphabet)
    # how many letters a word carries: adaptive counts, primed on the training pages
    sizes = Counter(min(len(group(w)), GROUP_CAP) for p in train_pages for ln in p.lines for w in ln.words)
    size_n = sum(sizes.values())
    # which word, among the words with the same reading: the judge's adaptive choice code
    pre = {}
    for u, c in base.train_counts.items():
        pre.setdefault(group(u), {})[u] = c
    state = {g: [dict(d), sum(d.values()), len(d)] for g, d in pre.items()}
    factor = FactorizedChoices(dec.key, base.train_counts, alphabet=alphabet,
                               normalize=lambda s: refcorpus.normalize(s, lang).replace(" ", ""))
    page_bits, plain_by_page = {}, {}
    factor_page_bits = {}
    letter_bits = size_bits = choice_bits = factor_choice_bits = 0.0
    for pg in eval_pages:
        stream, b_size, b_choice, b_factor = [], 0.0, 0.0, 0.0
        for ln in pg.lines:
            for w in ln.words:
                g = group(w)
                stream.append(g)
                if factor.supported:
                    b_factor += factor.bits(w, g, _literal_word_bits)
                k = min(len(g), GROUP_CAP)
                b_size -= math.log2((sizes[k] + 0.5) / (size_n + 0.5 * (GROUP_CAP + 1)))
                if k == GROUP_CAP:   # 4+: the exact length, Elias-gamma on the excess
                    b_size += 2 * (len(g) - GROUP_CAP + 1).bit_length() - 1
                sizes[k] += 1
                size_n += 1
                st = state.setdefault(g, [{}, 0, 0])
                counts_g, n_g, t_g = st
                c = counts_g.get(w, 0)
                if c:
                    b_choice -= math.log2(c / (n_g + t_g))
                else:
                    if n_g:
                        b_choice -= math.log2(t_g / (n_g + t_g))
                    b_choice += base.spell_bits(w)
                    st[2] = t_g + 1
                counts_g[w] = c + 1
                st[1] = n_g + 1
        letters = "".join(stream)
        b_letters = cm.bits(list(letters)) if letters else 0.0
        page_bits[pg.id] = b_letters + b_size + b_choice
        factor_page_bits[pg.id] = b_letters + b_size + b_factor
        plain_by_page[pg.id] = [[letters]] if letters else []
        letter_bits += b_letters
        size_bits += b_size
        choice_bits += b_choice
        factor_choice_bits += b_factor
    # A single flag selects the representation of lexical choices for the whole evaluation stream.
    choice_code = "adaptive_choice"
    if factor.supported and factor_choice_bits < choice_bits:
        choice_code = "factorized_preimage"
        choice_bits = factor_choice_bits
        page_bits = factor_page_bits
    info = {"plaintext_model": "spaceless_char_kn5", "plaintext_bits": round(letter_bits, 1),
            "letters_per_word_bits": round(size_bits, 1), "choice_code": choice_code,
            "choice_bits": round(choice_bits, 1)}
    return prog_bits + 2 + int(factor.supported), key_bits, page_bits, plain_by_page, info
