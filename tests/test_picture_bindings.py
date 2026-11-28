"""A visual verdict and an independently supported label measure different things."""
import unittest
from copy import deepcopy
from unittest.mock import patch

from vs import dsl, pictures, registry
from vs.ivtff import Page, Line


def obj(kind, label=None, **features):
    return {"kind": kind, "label": label, "features": features}


def figure(angle, label):
    return obj("human_figure", label, ring="outer", angle_deg_clockwise_from_top=angle,
               sex="female", posture="standing")


class BindingTest(unittest.TestCase):
    def test_same_drawing_with_erased_swapped_or_reordered_labels(self):
        a = [figure(359, "f1r.1"), figure(181, "f1r.2")]
        b = [figure(1, "f1r.1"), figure(179, "f1r.2")]

        def scores(right):
            pairs = pictures.compare_labels({"objects": a}, {"objects": right})
            return sorted((a[p.left]["label"], p.explicit, p.locus) for p in pairs)

        agreed = [("f1r.1", "same", "f1r.1"), ("f1r.2", "same", "f1r.2")]
        self.assertEqual(scores(b), agreed)
        def explicit_score():
            return pictures.explicit_label_agreement(pictures.compare_labels({"objects": a}, {"objects": b}))

        self.assertEqual(explicit_score(), 1)
        self.assertEqual(scores(b[::-1]), agreed)
        a.reverse()
        self.assertEqual(scores(b), agreed)
        b[0]["label"] = None
        self.assertEqual(scores(b), [("f1r.1", "one_missing", None), agreed[1]])
        self.assertEqual(explicit_score(), 0.5)
        b[0]["label"], b[1]["label"] = "f1r.2", "f1r.1"
        self.assertEqual(scores(b), [(lab, "different", None) for lab, _, _ in agreed])
        self.assertEqual(explicit_score(), 0)

    def test_unique_non_circular_kind_and_ambiguous_twins(self):
        a = [obj("plant", "f1r.1"), obj("jar", "f1r.2"), obj("jar", "f1r.3"), obj("sun")]
        b = deepcopy(a[::-1])
        pairs = pictures.compare_labels({"objects": a}, {"objects": b})
        self.assertEqual([(p.left, p.right, p.explicit, p.locus) for p in pairs],
                         [(0, 3, "same", "f1r.1"), (3, 0, "both_missing", None)])
        self.assertIsNone(pictures.explicit_label_agreement(pairs[1:]))

    def test_duplicate_locus_in_either_annotation_is_unusable(self):
        a = {"objects": [obj("plant", "f1r.1"), obj("jar", "f1r.2")]}
        b = {"objects": [obj("plant", "f1r.1"), obj("jar", "f1r.1")]}
        for left, right in ((a, b), (b, a)):
            self.assertFalse(any(p.locus for p in pictures.compare_labels(left, right)))

    def test_two_positions_can_fill_a_blank_only_if_no_explicit_locus_contradicts_them(self):
        a = {"objects": [figure(1, "f1r.1")]}
        b = {"objects": [figure(359, None)]}
        one = {0: "f1r.1"}
        for left, right, expected in ((one, one, "f1r.1"), ({}, one, None), (one, {}, None),
                                      (one, {0: "f1r.2"}, None), ({0: "f1r.2"}, {0: "f1r.2"}, None)):
            with self.subTest(left=left, right=right):
                self.assertEqual(pictures.compare_labels(a, b, left, right)[0].locus, expected)
        b["objects"].append(obj("jar", "f1r.1"))
        self.assertIsNone(pictures.compare_labels(a, b, one, one)[0].locus)

    def test_clock_fields_and_legacy_rows(self):
        for features, expected in (({"clock": "11:58"}, 359), ({"clock_position": "1-2"}, None),
                                   ({"bearing_clockwise_deg": -1}, 359),
                                   ({"position": "inner ring, 3 of 10 clockwise from the top"}, None),
                                   ({"position": "inner ring (about 2.4 o'clock)"}, 72)):
            self.assertEqual(pictures.clock_angle(obj("star", **features)), expected)
        a = {"statement": "Two rows", "objects": [dict(obj("jar"), id="row1-jar"), dict(obj("jar"), id="row2-jar")]}
        b = {"statement": "One jar in each row", "objects": [dict(obj("jar"), id="r2-pot"), dict(obj("jar"), id="r1-pot")]}
        self.assertEqual([(p.left, p.right) for p in pictures.compare_labels(a, b)], [(0, 1), (1, 0)])

    def test_tied_angles_different_rings_and_distant_objects(self):
        a = [figure(0, "f1r.1")]
        for b in ([figure(5, "f1r.1"), figure(355, None)], [figure(90, "f1r.1")],
                  [obj("human_figure", "f1r.1", ring="inner", clock_position=12)]):
            with self.subTest(b=b):
                self.assertEqual(pictures.compare_labels({"objects": a}, {"objects": b}), [])

    def test_an_unlocated_competitor_cannot_lose_a_clock_match(self):
        a = {"objects": [figure(0, "f1r.1")]}
        b = {"objects": [figure(25, "f1r.1"), obj("human_figure", ring="outer")]}
        self.assertEqual(pictures.compare_labels(a, b), [])
        self.assertEqual(pictures.compare_labels(b, a), [])
        with patch.object(dsl, "LABEL_POSITIONS", {"f1r": [[("f1r.1", 25)]]}):
            self.assertEqual(dsl._joined_labels("f1r", b["objects"]), {})

    def test_single_ring_figure_requires_independent_angles(self):
        a = {"objects": [figure(0, "f1r.1")]}
        b = {"objects": [obj("human_figure", "f1r.1", ring="outer")]}
        for left, right in ((a, b), (b, a)):
            self.assertEqual(pictures.compare_labels(left, right), [])

    def test_unique_central_object_does_not_require_an_angle(self):
        a = {"objects": [obj("star", "f1r.1", ring="centre")]}
        b = {"objects": [obj("star", "f1r.1", ring="centre")]}
        self.assertEqual(pictures.compare_labels(a, b), [pictures.LabelPair(0, 0, "same", "f1r.1")])

    def test_distinct_visual_features_can_identify_non_circular_objects(self):
        red = obj("jar", "f1r.1", colour="red")
        blue = obj("jar", "f1r.2", colour="blue")
        a = {"objects": [red, blue]}
        b = {"objects": [deepcopy(blue), deepcopy(red)]}
        pairs = pictures.compare_labels(a, b)
        self.assertEqual([(p.left, p.right, p.locus) for p in pairs], [(0, 1, "f1r.1"), (1, 0, "f1r.2")])
        # An unrecorded colour cannot exclude the other candidate.
        b["objects"][0]["features"].clear()
        self.assertEqual(pictures.compare_labels(a, b), [])


class ConsumerTest(unittest.TestCase):
    def setUp(self):
        dsl._confirmed_pictures.cache_clear()
        self.addCleanup(dsl._confirmed_pictures.cache_clear)

    def test_one_seat_a_different_visual_pair_and_a_contrary_third_vote(self):
        def record(seat, label="f1r.1", kind="plant"):
            return {"provenance": {"seat": seat}, "verdict": "AGREED", "agreement": 1,
                    "detail": {"folio": "f1r", "objects": [obj(kind, label, colour="red")]}}

        selected = record("a")
        for peers, expected in (([record("b")], {0: "f1r.1"}), ([record("a"), record("")], {}),
                                ([record("b", kind="root")], {})):
            self.assertEqual(dsl._confirmed_labels("f1r", selected, peers), expected)
        # Keep the third seat visually eligible despite its contradictory explicit locus.
        aligned, contrary = record("b"), record("c", "f1r.2")
        for r in (selected, aligned, contrary):
            r["detail"]["objects"].extend(obj(k, f"f1r.{i}", colour="red") for i, k in
                                          enumerate(("jar", "root", "sun", "moon", "star"), 3))
        self.assertGreaterEqual(registry.picture_agreement(selected["detail"], contrary["detail"]), 0.8)
        self.assertNotIn(0, dsl._confirmed_labels("f1r", selected, [aligned, contrary]))

    def test_projection_ties_include_competing_objects_with_ambiguous_lines(self):
        positions = {"f1r": [[("f1r.1", 0), ("f1r.2", 10)]]}
        with patch.object(dsl, "LABEL_POSITIONS", positions):
            for objects in ([figure(5, None), figure(350, None)], [figure(1, None), figure(359, None)]):
                self.assertEqual(dsl._joined_labels("f1r", objects), {})
                self.assertEqual(dsl._joined_labels("f1r", objects[::-1]), {})

    def test_unmatched_figure_claiming_a_locus_prevents_confirmation(self):
        def record(seat, objects):
            return {"provenance": {"seat": seat}, "verdict": "AGREED", "agreement": 1,
                    "detail": {"folio": "f1r", "objects": objects}}

        north = figure(0, None)
        north["features"].update(colour="red", clothing="nude", headwear="none", holds="star")
        selected = record("a", [north])
        aligned = record("b", [deepcopy(north)])
        contrary = record("c", [deepcopy(north), figure(180, "f1r.1")])
        unknown = deepcopy(north)
        unknown["features"].pop("angle_deg_clockwise_from_top")
        unknown["label"] = "f1r.1"
        unlocated = record("d", [unknown])
        self.assertGreaterEqual(registry.picture_agreement(selected["detail"], contrary["detail"]), 0.8)
        self.assertGreaterEqual(registry.picture_agreement(selected["detail"], unlocated["detail"]), 0.8)
        with patch.object(dsl, "LABEL_POSITIONS", {"f1r": [[("f1r.1", 0)]]}):
            self.assertEqual(dsl._confirmed_labels("f1r", selected, [aligned]), {0: "f1r.1"})
            self.assertEqual(dsl._confirmed_labels("f1r", selected, [aligned, contrary]), {})
            self.assertEqual(dsl._confirmed_labels("f1r", selected, [aligned, unlocated]), {0: "f1r.1"})

    def test_statistic_and_null_use_the_same_corroborated_lines(self):
        class Reverse:
            def shuffle(self, items):
                items.reverse()

        # Two confirmations, one disputed explicit label, one single-position guess.
        a = [figure(359, "f1r.1"), figure(91, None), figure(181, "f1r.3"),
             obj("star", None, ring="inner", clock_position=9)]
        b = [figure(1, "f1r.1"), figure(89, None), figure(179, "f1r.8"), obj("star")]
        records = [{"detail": {"folio": "f1r", "objects": objects}, "provenance": {"seat": seat},
                    "verdict": "AGREED", "agreement": 1.0} for seat, objects in (("a", a), ("b", b))]
        # More shared features on all three figures give this pair a real visual pass.
        for objs in (a, b):
            for o in objs:
                o["features"].update(clothing="nude", headwear="none", colour="red", holds="star")
        self.assertGreaterEqual(registry.picture_agreement(*(r["detail"] for r in records)), 0.8)
        loci = {"f1r": [[("f1r.1", 0), ("f1r.2", 90), ("f1r.3", 180)], [("f1r.4", 270)]]}
        toks = [{"folio": "f1r", "lid": f"f1r.{i}", "unit": "L", "w": f"word{i}"} for i in (1, 2, 3, 4, 8)]
        with patch.object(registry, "read", return_value=records), patch.object(dsl, "LABEL_POSITIONS", loci):
            self.assertEqual([o[3] for o in dsl._picture_objects(toks)], ["f1r.1", "f1r.2"])
            self.assertEqual([t["w"] for t in dsl._shuffled(toks, "shuffle_picture_labels", Reverse())],
                             ["word2", "word1", "word3", "word4", "word8"])
            page = Page("f1r", 0, {}, [Line(t["lid"], "f1r", i, "L", "0", [t["w"]], False, False)
                                       for i, t in enumerate(toks, 1)])
            spec = {"statistic": {"picture_label": {"test": "feature_label_share", "min_objects": 10}},
                    "compare": [{"op": ">", "value": 0}]}
            with self.assertRaises(dsl.SpecError) as exc:
                dsl.evaluate(spec, [page])
            self.assertIn("2 independently confirmed object-label links on the scored pages; need at least 10", str(exc.exception))

    def test_refusal_distinguishes_missing_visual_agreement_links_and_words(self):
        page = Page("f1r", 0, {}, [Line("f1r.1", "f1r", 1, "L", "0", ["word"], False, False)])
        spec = {"statistic": {"picture_label": {"test": "same_object_label", "min_objects": 10}},
                "compare": [{"op": ">", "value": 0}]}
        for confirmed, reason in (({}, "none of the scored folios has an AGREED picture annotation"),
                                  ({"f1r": []}, "visual agreement exists, but no drawing has an unambiguous label line"),
                                  ({"f1r": [(obj("jar"), "f1r.2")]}, "only 0 of 1 confirmed loci have label words")):
            with self.subTest(reason=reason), patch.object(dsl, "_confirmed_pictures", return_value=confirmed):
                with self.assertRaises(dsl.SpecError) as exc:
                    dsl.evaluate(spec, [page])
                self.assertIn("0 independently confirmed object-label links", str(exc.exception))
                self.assertIn(reason, str(exc.exception))
