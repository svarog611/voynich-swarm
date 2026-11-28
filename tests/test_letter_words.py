"""Letters written as whole words: the family, the respaced reading, the control and the search."""
import os, random, unittest
from collections import Counter
from math import log2
from unittest.mock import patch
from vs import path, views
from vs.families import letter_words
from vs.letter_word_preimage import FactorizedChoices

HAVE_DATA = os.path.exists(path("data", "raw", "ZL3b-n.txt"))
KEY = {"prefix": {"qo": "a", "ch": "b", "q": "c"}, "suffix": {"dy": "d", "aiin": "e", "y": "f"}}


class FamilyTest(unittest.TestCase):
    def test_a_word_is_a_prefix_a_suffix_or_both(self):
        d = lambda w: letter_words.decode_word(w, KEY, None)
        self.assertEqual(d("qody"), "ad")
        self.assertEqual(d("ch"), "b")
        self.assertEqual(d("aiin"), "e")
        self.assertEqual(d("qy"), "cf", "the longest prefix whose remainder is a suffix")
        self.assertEqual(d("qoxx"), "", "a word that cannot be cut carries nothing")
        self.assertEqual(letter_words.units("qody", KEY), ["<qo", ">dy"])

    def test_only_this_family_is_respaced_and_its_key_has_two_tables(self):
        ok = {"family": "letter-words", "plaintext": "respaced", "key": KEY}
        views.check(ok)
        for bad in ({**ok, "plaintext": "words"}, {"family": "substitution", "plaintext": "respaced", "key": {"map": {}}},
                    {**ok, "key": {"map": {}}}, {**ok, "spaces": "ignored"}):
            with self.assertRaises(SystemExit):
                views.check(bad)

    def test_factorized_choices_exclude_noncanonical_reconstructions(self):
        key = {"prefix": {"q": "a", "qo": "a", "qr": "b"}, "suffix": {"r": "c"}}
        training = Counter({"qor": 4, "qr": 1, "opaque": 2})
        choice = FactorizedChoices(key, training)
        self.assertTrue(choice.supported)
        self.assertEqual(letter_words.decode_word("qr", key, None), "b")
        # Only "qor" can encode ac. Its cost is the trackable match flag.
        self.assertAlmostEqual(choice.bits("qor", "ac", lambda _: 11), -log2(5.5 / 8))
        # An unrecognised word must take a literal escape.
        self.assertAlmostEqual(choice.bits("opaque", "", lambda _: 11), 11 - log2(2.5 / 8))

    def test_factorized_choice_costs_form_a_normalized_distribution(self):
        key = {"prefix": {"q": "a", "qo": "a"}, "suffix": {"r": "b", "y": "b"}}
        training = Counter({"qr": 3, "qoy": 2, "qy": 1, "opaque": 1})
        choice = FactorizedChoices(key, training)
        regular = sum(2 ** -choice.bits(word, "ab", lambda _: 0)
                      for word in ("qr", "qy", "qor", "qoy"))
        escape = 2 ** -choice.bits("opaque", "ab", lambda _: 0)
        self.assertAlmostEqual(regular + escape, 1.0)

    def test_noncanonical_output_uses_the_existing_mode(self):
        self.assertFalse(FactorizedChoices({"prefix": {"q": "ae"}, "suffix": {}}, {}).supported)

    def test_optional_preimage_skips_excessive_enumeration(self):
        key = {"prefix": {"q" * 4000: "a"}, "suffix": {"r": "b"}}
        self.assertFalse(FactorizedChoices(key, {}).supported)

    def test_visible_scoring_uses_preimages_for_new_sign_combinations(self):
        from vs import ivtff, judge, refcorpus, respace

        def page(name, words):
            line = ivtff.Line(name + ".1", name, 1, "P", "0", words, True, True)
            return ivtff.Page(name, 0, {}, [line])

        key = {"prefix": dict.fromkeys(("q", "qo", "k", "t"), "a"),
               "suffix": dict.fromkeys(("d", "r", "y", "m"), "b")}
        sub = {"family": "letter-words", "plaintext": "respaced", "key": key}
        train = [page("f1r", ["qd", "qor", "ky", "tm"] * 16)]
        evaluation = [page("f1v", ["qr", "qoy", "km", "td"] * 4)]
        decoder = judge.Decoder(sub, "french")
        baseline = judge.Baseline(train, decoder.seg)
        with patch.object(refcorpus, "train_text", return_value=["abab ab aba"]):
            program, _, per_page, plain, info = respace.description_length(
                decoder, baseline, train, evaluation, "french", judge.PARAMS
            )
        self.assertEqual(info["choice_code"], "factorized_preimage")
        self.assertAlmostEqual(info["choice_bits"], round(16 * (4 + log2(65 / 64.5)), 1))
        self.assertEqual(plain["f1v"], [["ab" * 16]])
        self.assertGreater(per_page["f1v"], 0)
        self.assertEqual(program, judge.program_bits(letter_words.__file__) + 3)


@unittest.skipUnless(HAVE_DATA, "run python3 -m vs fetch first")
class RespacedTest(unittest.TestCase):
    def test_the_cut_finds_the_words_of_a_stream(self):
        from vs import respace
        self.assertEqual(respace.cut("inprincipioeratuerbum", "latin"), ["in", "principio", "erat", "uerbum"])
        m = respace.metrics("latin", ["inprincipioeratuerbumetuerbumeratapuddeum"])
        self.assertGreater(m["cov"], 0.8)
        rng = random.Random(1)
        noise = "".join(rng.choice("abcdefghilmnopqrstux") for _ in range(400))
        self.assertLess(respace.metrics("latin", [noise])["cov"], 0.5)
        self.assertGreater(respace.metrics("latin", [noise])["H"], m["H"] + 1.0)

    def test_the_search_recovers_the_planted_key_at_two_percent_noise(self):
        from vs import controls, search
        tr, _ev, true = controls.make("latin", "letter-words", noise=controls.NOISE)
        found = search.search_cell("latin", "letter-words", tr, tag="test")
        self.assertEqual(found["plaintext"], "respaced")
        self.assertEqual(found["search"]["front_glyphs"], sorted(["q", "o", "k", "t", "p", "f", "ch", "sh", "cth", "ckh"]))
        count, right = Counter(), Counter()
        for p in tr:
            for ln in p.lines:
                for w in ln.words:
                    count[w] += 1
        seg = None
        same = sum(n for w, n in count.items()
                   if letter_words.decode_word(w, found["key"], seg) == letter_words.decode_word(w, true["key"], seg))
        self.assertGreater(same / sum(count.values()), 0.97)

    def test_the_true_key_passes_and_a_scrambled_key_fails(self):
        from vs import controls, judge
        tr, ev, true = controls.make("latin", "letter-words", noise=controls.NOISE)
        r = judge.evaluate(true, tr, ev, "control")
        self.assertTrue(r["a_compresses"]["pass"] and r["b_reads"]["pass"] and r["kill_tests"]["pass"], r["failed"])
        self.assertTrue(r["d_not_illusion"]["pass"], r["d_not_illusion"])
        rng = random.Random(7)
        wrong = {**true, "key": {t: dict(zip(m, rng.sample(list(m.values()), len(m)))) for t, m in true["key"].items()}}
        bad = judge.evaluate(wrong, tr, ev, "control")
        self.assertFalse(bad["a_compresses"]["pass"])
        self.assertFalse(bad["b_reads"]["pass"])


if __name__ == "__main__":
    unittest.main()
