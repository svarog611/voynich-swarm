"""One drawing under different names (plant / root / plant_part; jar / container) still pairs, and
ring folios ask for a measured angle until two seats have given one."""
import unittest
from unittest.mock import patch

from vs import dsl, ivtff, pictures, registry, steps


def obj(i, kind, label=None, **features):
    return {"id": str(i), "kind": kind, "label": label, "features": features}


def note(*objects):
    return {"folio": "f99v", "objects": list(objects)}


class KindClassTest(unittest.TestCase):
    def test_plant_parts_are_one_kind_and_jars_are_containers(self):
        for kind in ("plant", "plant_part", "root", "leaf", "flower"):
            self.assertEqual(pictures.kind_class(kind), "plant")
        self.assertEqual(pictures.kind_class("container"), pictures.kind_class("jar"))
        for kind in ("star", "human_figure", "pool", "other", None):
            self.assertEqual(pictures.kind_class(kind), kind)

    def test_a_different_name_for_the_same_drawing_does_not_leave_it_unpaired(self):
        a = note(obj(1, "root", colour="red"), obj(2, "jar", colour="blue"))
        b = note(obj(1, "plant_part", colour="red"), obj(2, "container", colour="blue"))
        self.assertEqual(len(pictures.pair_objects(a["objects"], b["objects"])), 2)
        self.assertEqual(pictures.agreement(a, b), 1.0)

    def test_what_the_seats_see_is_still_compared(self):
        a = note(obj(1, "root", colour="red"))
        b = note(obj(1, "plant", colour="green"))
        self.assertEqual(pictures.agreement(a, b), 0.0)

    def test_different_things_stay_unpaired_and_count_against(self):
        a = note(obj(1, "root", colour="red"))
        b = note(obj(1, "star", colour="red"))
        self.assertEqual(pictures.pair_objects(a["objects"], b["objects"]), [])
        self.assertEqual(pictures.agreement(a, b), 0.0)

    def test_kind_named_differently_under_one_label_agrees(self):
        a = note(obj(1, "leaf", "f99v.3", colour="green"))
        b = note(obj(1, "plant", "f99v.3", colour="green"))
        self.assertEqual(pictures.agreement(a, b), 1.0)


class MeasuredAngleTest(unittest.TestCase):
    def test_only_a_number_of_degrees_is_a_measurement(self):
        self.assertEqual(pictures.measured_angle(obj(1, "human_figure", angle_deg_clockwise_from_top=365)), 5.0)
        self.assertEqual(pictures.measured_angle(obj(1, "human_figure", bearing_clockwise_deg=12.5)), 12.5)
        for features in ({"clock_position": 3}, {"position": "outer ring, 3 of 20 clockwise from the top (about 1.2 o'clock)"},
                         {"angle_deg_clockwise_from_top": "90"}, {"angle_deg_clockwise_from_top": True}, {}):
            self.assertIsNone(pictures.measured_angle(obj(1, "human_figure", **features)))


class RingFolioStepTest(unittest.TestCase):
    def pic_steps(self, records):
        line = ivtff.Line("f72v1.2", "f72v1", 2, "L", "0", ["otaly"], True, True)
        pages = [ivtff.Page("f72v1", 0, {}, [line])]
        read = lambda type=None, **_: records if type == "picture" else []
        with patch.object(registry, "read", side_effect=read), patch.object(ivtff, "parse", return_value=pages), \
                patch.object(dsl, "label_positions", return_value={"f72v1": [[("f72v1.2", 0.0)]]}):
            return [s for s in steps.derived_steps() if s["id"] == "PIC-f72v1"]

    def record(self, seat, **features):
        return {"id": "V-" + seat, "type": "picture", "verdict": "AGREED", "provenance": {"seat": seat},
                "detail": {"folio": "f72v1", "objects": [obj(1, "human_figure", ring="outer", **features)]}}

    def test_two_annotations_without_two_measured_angles_reopen_the_folio(self):
        got = self.pic_steps([self.record("a", angle_deg_clockwise_from_top=10), self.record("b", clock_position=12)])
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["added_by"], "rule:ring-folio-angles")
        self.assertIn("measured", got[0]["title"])

    def test_two_measuring_seats_close_it(self):
        self.assertEqual(self.pic_steps([self.record("a", angle_deg_clockwise_from_top=10),
                                         self.record("b", angle_deg_clockwise_from_top=12)]), [])

    def test_one_seat_measuring_twice_is_one_seat(self):
        got = self.pic_steps([self.record("a", angle_deg_clockwise_from_top=10), self.record("a", angle_deg_clockwise_from_top=12),
                              self.record("b", clock_position=12)])
        self.assertEqual(len(got), 1)


if __name__ == "__main__":
    unittest.main()
