"""Picture-text statistics (vs/dsl.py picture_label, null shuffle_picture_labels), step P-L4-013.
Synthetic pages and picture annotations only: nothing here reads the record or the manuscript."""
import random, unittest
from copy import deepcopy
from unittest.mock import patch
from vs import dsl, registry
from vs.ivtff import Page, Line

VOCAB = ["daiin", "chol", "shedy", "qokeedy", "okal", "otedy", "chedy", "qokain", "sheol", "dalor",
         "kedar", "pchol", "tshey", "olkeey", "ykedy", "rotal", "sairy", "cphor", "ytor", "lkain"]


def make(rng, n_pages=12, labels_per_page=3, label_from_text=True):
    """Pages with running text and label lines; returns pages and a picture annotation per folio."""
    pages, pics = [], {}
    for i in range(n_pages):
        pid = f"f{i}r"
        p = Page(pid, i, {"I": "Z", "L": "A", "H": "1"})
        own = [f"{rng.choice(['ot', 'ok', 'qo', 'ch'])}{rng.choice(['al', 'ar', 'ey', 'ol'])}{i}x{j}" for j in range(labels_per_page)]
        n = 1
        for _ in range(8):
            ws = [rng.choice(VOCAB) for _ in range(8)]
            if label_from_text:
                ws[0] = rng.choice(own)
            p.lines.append(Line(f"{pid}.{n}", pid, n, "P", "0", ws, n == 1, False)); n += 1
        objs = []
        for j, w in enumerate(own):
            lab = w if label_from_text else f"zz{i}y{j}"
            p.lines.append(Line(f"{pid}.{n}", pid, n, "L", "0", [lab], False, False))
            objs.append({"id": f"o{j}", "kind": ["plant", "jar", "star"][j % 3],
                         "features": {"shape": ["round", "tall", "spiky"][j % 3], "colour": rng.choice(["green", "blue"])},
                         "label": f"{pid}.{n}"})
            n += 1
        pages.append(p)
        pics[pid] = objs
    return pages, pics


def spec(test, null=True, perms=99, p=0.05):
    c = [{"null": "shuffle_picture_labels", "perms": perms, "p": p}] if null else [{"op": ">", "value": 0}]
    return {"statistic": {"picture_label": {"test": test, "min_objects": 10}}, "compare": c}


class PictureLabelTest(unittest.TestCase):
    def use_pictures(self, pics):
        records = [{"detail": {"folio": folio, "objects": deepcopy(objects)}, "provenance": {"seat": seat},
                    "verdict": "AGREED", "agreement": 1.0}
                   for folio, objects in pics.items() for seat in ("a", "b")]
        self.enterContext(patch.object(registry, "read", return_value=records))
        dsl._confirmed_pictures.cache_clear()
        self.addCleanup(dsl._confirmed_pictures.cache_clear)

    def tearDown(self):
        dsl.LABEL_POSITIONS = None

    def test_shape(self):
        with self.assertRaises(dsl.SpecError):
            dsl.check({"statistic": {"picture_label": {"test": "nonsense"}}, "compare": [{"op": ">", "value": 0}]})
        with self.assertRaises(dsl.SpecError):   # too few objects allowed
            dsl.check({"statistic": {"picture_label": {"test": "label_in_page_text", "min_objects": 3}}, "compare": [{"op": ">", "value": 0}]})
        with self.assertRaises(dsl.SpecError):   # unknown key
            dsl.check({"statistic": {"picture_label": {"test": "label_in_page_text", "folios": ["f1r"]}}, "compare": [{"op": ">", "value": 0}]})

    def test_refused_without_enough_objects(self):
        pages, pics = make(random.Random(1), n_pages=2)
        self.use_pictures(pics)       # 6 confirmed labelled objects < 10
        with self.assertRaises(dsl.SpecError):
            dsl.evaluate(spec("label_in_page_text", null=False), pages)

    def test_label_in_page_text_found_and_beats_null(self):
        pages, pics = make(random.Random(2))
        self.use_pictures(pics)
        r = dsl.evaluate(spec("label_in_page_text"), pages)
        self.assertGreater(r["value"], 0.5)
        self.assertTrue(r["pass"], r)

    def test_label_in_page_text_absent_does_not_pass(self):
        pages, pics = make(random.Random(3), label_from_text=False)
        self.use_pictures(pics)
        r = dsl.evaluate(spec("label_in_page_text", null=False), pages)
        self.assertLessEqual(r["value"], 0)
        self.assertFalse(r["pass"])

    def test_objects_off_the_scored_pages_do_not_count(self):
        pages, pics = make(random.Random(4))
        pics = dict(pics, f99v=[{"id": "x", "kind": "jar", "features": {}, "label": "f99v.1"}])
        self.use_pictures(pics)
        self.assertEqual(len(dsl._picture_objects(dsl.tokens(pages))), 36)

    def test_null_keeps_running_text_and_label_words(self):
        pages, pics = make(random.Random(5))
        self.use_pictures(pics)
        toks = dsl.tokens(pages)
        sh = dsl._shuffled(toks, "shuffle_picture_labels", random.Random(0))
        run = lambda ts: [(t["lid"], t["w"]) for t in ts if t["unit"] == "P"]
        self.assertEqual(run(toks), run(sh))
        lab = lambda ts: sorted(t["w"] for t in ts if t["unit"] == "L")
        self.assertEqual(lab(toks), lab(sh))
        self.assertNotEqual([t["w"] for t in toks if t["unit"] == "L"], [t["w"] for t in sh if t["unit"] == "L"])

    def test_same_object_label(self):
        rng = random.Random(6)
        pages, pics = make(rng, label_from_text=False)
        # the same-looking jar on every folio gets one label (or one glyph off): the statistic must see it
        for i, p in enumerate(pages):
            for ln in p.lines:
                if ln.id == pics[p.id][1]["label"]:
                    ln.words = ["okeody" if i % 2 else "okeedy"]
            pics[p.id][1]["features"] = {"shape": "tall", "colour": "blue", "lid": "flat"}
        self.use_pictures(pics)
        r = dsl.evaluate(spec("same_object_label"), pages)
        self.assertGreater(r["value"], 0.5)
        self.assertTrue(r["pass"], r)

    def test_feature_label_share_defined(self):
        pages, pics = make(random.Random(7))
        self.use_pictures(pics)
        v = dsl.statistic({"picture_label": {"test": "feature_label_share", "min_objects": 10}}, [], dsl.tokens(pages))
        self.assertTrue(v == v)

    def test_layout_bookkeeping_is_not_a_look(self):
        o = {"features": {"ring": "outer", "clock_position": 3, "label_ring_candidates": "x", "colour": "Blue "}}
        self.assertEqual(dsl._looks(o), {"colour": "blue"})


    def test_join_by_clock_angle(self):
        dsl.LABEL_POSITIONS = {"f9r": [[("f9r.2", 0.0), ("f9r.3", 90.0), ("f9r.4", 180.0)], [("f9r.6", 45.0)]]}
        objs = [{"kind": "human_figure", "features": {"ring": "outer", "angle_deg_clockwise_from_top": 85}},
                {"kind": "human_figure", "features": {"ring": "outer", "angle_deg_clockwise_from_top": 350}},
                {"kind": "human_figure", "features": {"ring": "inner", "clock_position": 2}},       # 60 deg, 15 from f9r.6
                {"kind": "human_figure", "features": {"ring": "outer", "angle_deg_clockwise_from_top": 130}},  # 40 deg off: too far
                {"kind": "star", "features": {"ring": "outer", "angle_deg_clockwise_from_top": 180}, "label": "f9r.4"}]  # already labelled
        self.assertEqual(dsl._joined_labels("f9r", objs), {0: "f9r.3", 1: "f9r.2", 2: "f9r.6", 4: "f9r.4"})

    def test_join_gives_a_line_to_the_closest_object_only(self):
        dsl.LABEL_POSITIONS = {"f9r": [[("f9r.2", 0.0), ("f9r.3", 180.0)]]}
        objs = [{"kind": "human_figure", "features": {"ring": "outer", "angle_deg_clockwise_from_top": 10}},
                {"kind": "human_figure", "features": {"ring": "outer", "angle_deg_clockwise_from_top": 355}}]
        self.assertEqual(dsl._joined_labels("f9r", objs), {1: "f9r.2"})

    def test_ring_names_follow_the_number_of_rings(self):
        dsl.LABEL_POSITIONS = {"f9r": [[("f9r.2", 0.0)], [("f9r.4", 0.0)], [("f9r.6", 0.0)]]}
        objs = [{"kind": "human_figure", "features": {"ring": "middle", "angle_deg_clockwise_from_top": 5}},
                {"kind": "human_figure", "features": {"ring": "inner", "angle_deg_clockwise_from_top": 5}}]
        self.assertEqual(dsl._joined_labels("f9r", objs), {0: "f9r.4", 1: "f9r.6"})


    def test_experiment_checker_accepts_a_picture(self):
        """vs experiment used to fall through to the segmentation scorer on kind picture (KeyError 'units')."""
        import json, os, tempfile, io, contextlib
        from vs import experiment
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "experiments", "PIC-f1r", "t")
            os.makedirs(d)
            json.dump({"step": "PIC-f1r", "handle": "t", "kind": "picture", "summary": "s", "lease": "x"}, open(os.path.join(d, "experiment.json"), "w"))
            json.dump({"folio": "f1r", "canvas": "c", "blind": True,
                       "objects": [{"id": "a", "kind": "plant", "features": {}, "label": "f1r.1"}]}, open(os.path.join(d, "picture.json"), "w"))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(experiment.main(d), 0)
            self.assertIn("1 with a label locus", out.getvalue())


if __name__ == "__main__":
    unittest.main()
