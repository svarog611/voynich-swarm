"""A fact that cannot be measured on an attacker's text is not reproduced by it (vs/redteam.py).

Before, the scorer raised: one fact undefined on the synthetic text made a whole generator test a
FAULT, which is retried on every sweep (#825 and #844, 28 Nov 2026)."""
import unittest
from unittest.mock import patch

from vs import corpus, dsl, redteam

FACT = {"id": "F-900", "type": "fact", "verdict": "ESTABLISHED", "detail": {"spec": {"statistic": {"x": 1}}}}
OTHER = {"id": "F-901", "type": "fact", "verdict": "ESTABLISHED", "detail": {"spec": {"statistic": {"y": 1}}}}


def lines():
    _train, ev = corpus.split_pages("practice")
    need = sum(1 for p in ev for ln in p.lines if ln.words)
    widest = max(len(ln.words) for p in ev for ln in p.lines)
    return [[f"zz{i}x{j}" for j in range(widest)] for i in range(need)]


class UndefinedOnSyntheticText(unittest.TestCase):
    def setUp(self):
        _train, ev = corpus.split_pages("practice")
        self.real_pages = ev

    def evaluate(self, undefined_for):
        def fake_evaluate(spec, pages, train=None):
            if pages is not self.real_pages and spec in undefined_for:
                raise dsl.SpecError("the statistic is undefined on this selection (3 tokens)")
            return {"pass": True, "value": 1.0}
        return fake_evaluate

    def run_with(self, undefined_for, call):
        with patch.object(corpus, "split_pages", return_value=([], self.real_pages)), \
                patch.object(dsl, "evaluate", side_effect=self.evaluate(undefined_for)):
            return call()

    def test_a_fact_attack_fails_instead_of_raising(self):
        r = self.run_with([FACT["detail"]["spec"]], lambda: redteam.fact_attack("F-900", lines(), "practice", FACT))
        self.assertFalse(r["success"])
        self.assertNotIn("refused", r)
        self.assertEqual(r["synthetic"]["pass"], False)
        self.assertIn("undefined", r["synthetic"]["undefined"])

    def test_a_generator_test_counts_it_as_not_reproduced_and_scores_the_rest(self):
        r = self.run_with([FACT["detail"]["spec"]],
                          lambda: redteam.generator_test(lines(), "practice", facts_on_record=[FACT, OTHER]))
        self.assertEqual(r["per_fact"], {"F-900": False, "F-901": True})
        self.assertEqual((r["reproduced"], r["of"]), (1, 2))

    def test_a_fact_undefined_on_the_manuscript_is_still_an_error(self):
        def always(spec, pages, train=None):
            raise dsl.SpecError("undefined")
        with patch.object(corpus, "split_pages", return_value=([], self.real_pages)), patch.object(dsl, "evaluate", side_effect=always):
            with self.assertRaises(dsl.SpecError):
                redteam.fact_attack("F-900", lines(), "practice", FACT)


if __name__ == "__main__":
    unittest.main()
