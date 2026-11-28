import unittest
from unittest.mock import patch

from vs import controls, ivtff, judge, search, season
from vs.control_recovery import letter_words_recovery


PLANTED = {"prefix": {"p": "a", "q": "c"}, "suffix": {"l": "e", "n": "x"}}
FITTED = {"prefix": {"p": "a"}, "suffix": {"l": "f"}}


def pages(name, words):
    line = ivtff.Line(name + ".1", name, 1, "P", "0", words, True, True)
    return [ivtff.Page(name, 0, {}, [line])]


class LetterWordsRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.clean = (pages("f1r", ["pl", "pl"]), pages("f1v", ["pl", "ql"]))
        self.noisy = (pages("f1r", ["pl", "qn"]), pages("f1v", ["pn", "ql"]))

    def test_distinguishes_clean_omission_from_noisy_replacement(self):
        report = letter_words_recovery(PLANTED, FITTED, self.clean, self.noisy)
        self.assertEqual(report["roles"]["prefix"]["correct_fitted_signs"], 1)
        self.assertEqual(report["roles"]["suffix"]["correct_fitted_signs"], 0)
        self.assertEqual(report["training"]["omitted_clean_sign_occurrences"], 0)
        self.assertEqual(report["training"]["omitted_received_occurrences_at_changed_positions"], 2)
        self.assertEqual(report["evaluation"], {
            "ciphertext_groups": 2, "exact_against_clean": 0, "exact_against_received": 0,
            "omitted_received_sign_types": 2, "omitted_clean_sign_occurrences": 1,
            "omitted_received_sign_occurrences": 2,
            "omitted_received_occurrences_without_clean_training_support": 2,
            "omitted_received_occurrences_at_changed_positions": 1,
        })

    def test_perfect_key_reports_noise_separately_from_key_recovery(self):
        report = letter_words_recovery(PLANTED, PLANTED, self.clean, self.noisy)
        self.assertEqual(report["evaluation"]["exact_against_clean"], 1)
        self.assertEqual(report["evaluation"]["exact_against_received"], 2)

    def test_calibration_reports_recovery_in_both_noise_arms(self):
        mk = [(*self.clean, {"key": PLANTED}), (*self.noisy, {"key": PLANTED})]
        with patch.object(controls, "make", side_effect=mk), \
                patch.object(search, "search_cell", return_value={"key": FITTED}), \
                patch.object(judge, "evaluate", return_value={"passes_without_anchors": False}), \
                patch.object(season, "_short", side_effect=lambda result: result):
            row = season.calibrate_cell("french", "letter-words")
        self.assertFalse(row["search_cracks_control"])
        self.assertEqual(set(row["letter_words_recovery"]), {"clean", "noisy"})
        clean, noisy = (row["letter_words_recovery"][kind]["training"] for kind in ("clean", "noisy"))
        self.assertEqual(clean["omitted_received_sign_occurrences"], 0)
        self.assertEqual(noisy["omitted_received_sign_occurrences"], 2)


if __name__ == "__main__":
    unittest.main()
