"""A factorized ciphertext choice code for canonical letter-words keys."""

import math
from collections import Counter, defaultdict
from itertools import product

from .families import letter_words

MAX_ENUMERATION_WORK = 10_000_000


class FactorizedChoices:
    """Encode a raw word conditional on its known decoded group.

    Each token pays for a Bernoulli escape flag learned on training words. The
    regular branch chooses among exact preimages, weighted by the product of
    training counts (with half-count smoothing) of the nonempty role signs.
    The escape branch uses a complete literal word code supplied by the caller.
    The caller also accounts for the stream-wide choice of coding mode.
    """

    def __init__(self, key, training_words, *, normalize=None, alphabet=None):
        self._normalize = normalize
        self._key = key
        allowed = set(alphabet) if alphabet is not None else None
        tables = [key.get(role) for role in ("prefix", "suffix")] if isinstance(key, dict) else []
        self.supported = len(tables) == 2 and all(
            isinstance(table, dict) and all(
                isinstance(sign, str) and sign and not any(c.isspace() for c in sign)
                and isinstance(letter, str) and len(letter) == 1 and letter.isalpha()
                and (allowed is None or letter in allowed)
                and (normalize is None or normalize(letter) == letter)
                for sign, letter in table.items()
            ) for table in tables
        )
        if not self.supported:
            return
        prefix, suffix = tables
        combinations = (len(prefix) + 1) * (len(suffix) + 1) - 1
        longest = max(map(len, prefix), default=0) + max(map(len, suffix), default=0)
        # The canonical splitter slices at each offset. Keep the optional code
        # bounded even for tables with many unobserved signs or very long signs.
        if combinations * longest ** 2 > MAX_ENUMERATION_WORK:
            self.supported = False
            return

        self._inverse = [defaultdict(list), defaultdict(list)]
        self._counts = [Counter(), Counter()]
        self._groups = {}
        for inverse, table in zip(self._inverse, tables):
            for sign, letter in sorted(table.items()):
                inverse[letter].append(sign)

        hits = misses = 0
        for word, frequency in Counter(training_words).items():
            cut = letter_words.split(word, key)
            if cut and any(cut):
                decoded = "".join(table.get(sign, "") for table, sign in zip(tables, cut))
                if normalize is None or normalize(decoded) == decoded:
                    hits += frequency
                    for counts, sign in zip(self._counts, cut):
                        if sign:
                            counts[sign] += frequency
                    continue
            misses += frequency
        total = hits + misses + 1
        self._match_bits = math.log2(total / (hits + 0.5))
        self._escape_bits = math.log2(total / (misses + 0.5))

    def _group(self, decoded):
        if decoded in self._groups:
            return self._groups[decoded]

        pre, suf = self._inverse
        cuts = ()
        if self._normalize is None or self._normalize(decoded) == decoded:
            if len(decoded) == 1:
                cuts = [(p, "") for p in pre.get(decoded, ())]
                cuts.extend(("", s) for s in suf.get(decoded, ()))
            elif len(decoded) == 2:
                cuts = product(pre.get(decoded[0], ()), suf.get(decoded[1], ()))
        weights = {}
        for p, s in cuts:
            word = p + s
            if letter_words.split(word, self._key) == (p, s):
                weights[word] = math.prod(
                    counts[sign] + 0.5 for counts, sign in zip(self._counts, (p, s)) if sign
                )
        result = weights, math.fsum(weights.values())
        self._groups[decoded] = result
        return result

    def bits(self, raw_word, decoded, literal_bits):
        """Return the ideal code length, calling literal_bits(raw_word) only on escape."""
        if not self.supported:
            return literal_bits(raw_word)
        weights, total = self._group(decoded)
        weight = weights.get(raw_word)
        if weight is None:
            return self._escape_bits + literal_bits(raw_word)
        return self._match_bits + math.log2(total / weight)
