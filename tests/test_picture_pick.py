import unittest
from unittest.mock import patch

from vs import dsl, registry


FOLIO = "f72v1"


def annotation(record_id, agreement, labels):
    objects = [
        {"id": f"figure-{index}", "kind": "human_figure", "label": label}
        for index, label in enumerate(labels, 1)
    ]
    return {
        "id": record_id,
        "verdict": "AGREED",
        "agreement": agreement,
        "detail": {"folio": FOLIO, "objects": objects},
    }


class PicturePickTest(unittest.TestCase):
    def assert_selected_in_both_orders(self, first, second, expected):
        for records in ([first, second], [second, first]):
            with self.subTest(first=records[0]["id"]), patch.object(registry, "read", return_value=records):
                self.assertEqual(dsl.pictures(), {FOLIO: expected["detail"]["objects"]})

    def test_f72v1_tie_uses_annotation_with_explicit_loci(self):
        loci = [f"{FOLIO}.{line}" for line in (*range(2, 22), *range(23, 33))]
        ring_only = annotation("V-026", 0.833, [None] * 32)
        explicit = annotation("V-043", 0.833, [None, *loci, None])
        self.assert_selected_in_both_orders(ring_only, explicit, explicit)

    def test_superior_agreement_wins_with_fewer_loci(self):
        higher = annotation("V-high", 0.834, [None])
        lower = annotation("V-low", 0.833, [f"{FOLIO}.2", f"{FOLIO}.3"])
        self.assert_selected_in_both_orders(higher, lower, higher)

    def test_duplicate_blank_and_nonstring_labels_do_not_count(self):
        first_locus, second_locus = f"{FOLIO}.2", f"{FOLIO}.3"
        explicit = annotation("V-valid", 0.833, [first_locus, second_locus])
        ignored = {
            "duplicate": [first_locus, first_locus],
            "blank": ["", " ", "\t"],
            "nonstring": [42, [second_locus], {"locus": second_locus}],
        }
        for case, labels in ignored.items():
            with self.subTest(case=case):
                inflated = annotation("V-inflated", 0.833, [first_locus, *labels])
                self.assert_selected_in_both_orders(inflated, explicit, explicit)


if __name__ == "__main__":
    unittest.main()
