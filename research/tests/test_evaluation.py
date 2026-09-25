"""Synthetic tests only: no photos, embeddings, or weights enter Git."""

from __future__ import annotations

from io import BytesIO
import sys
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fckface_lab.calibration import CalibrationPair, calibrate
from fckface_lab.evaluation import (
    GalleryEvaluation, ModelPhotoResults, PhotoResult, Reference,
    development_report, evaluate_gallery, release_gate,
)
from fckface_lab.imaging import decode_image, export_jpeg, make_variants
from fckface_lab.recognition import (
    EmbeddingResult, detector_frame, restore_face_coordinates, select_face,
)


MATCH = GalleryEvaluation("valid", {"own": .9}, ("own",), .9, True)
NONMATCH = GalleryEvaluation("valid", {"own": .1}, (), .1, False)
INCONCLUSIVE = GalleryEvaluation("inconclusive", reason="no_face")
MODELS = ("AdaFace", "MagFace", "EdgeFace", "TransFace")
KEYS = (("export", "one"), ("blur", "one"))


def photo(photo_id="p", *, controls=None, altered=None):
    controls = controls or {}
    altered = altered or {}
    return PhotoResult(photo_id, {
        model: ModelPhotoResults(
            {key: controls.get((model, key), MATCH) for key in KEYS},
            {key: altered.get((model, key), NONMATCH) for key in KEYS
             if (model, key) not in altered or altered[(model, key)] is not None},
        ) for model in MODELS
    })


class GateTests(unittest.TestCase):
    def gate(self, items):
        return release_gate(items, conditions=("export", "blur"), galleries=("one",))

    def test_three_models_every_condition_passes_and_two_fails(self):
        passed = photo(altered={("AdaFace", KEYS[1]): MATCH})
        failed = photo("failed", altered={("AdaFace", KEYS[1]): MATCH,
                                          ("MagFace", KEYS[0]): MATCH})
        result = self.gate((passed, failed))
        self.assertEqual((result.eligible_count, result.successful_count), (2, 1))
        self.assertFalse(result.passed)  # 50% is below the 95% gate
        self.assertEqual(result.successful_photo_ids, ("p",))

    def test_missing_altered_is_failure_and_missing_control_is_ineligible(self):
        missing_altered = photo(altered={("AdaFace", KEYS[0]): None,
                                         ("MagFace", KEYS[0]): None})
        missing_control = photo("control", controls={("AdaFace", KEYS[0]): INCONCLUSIVE})
        result = self.gate((missing_altered, missing_control))
        self.assertEqual(result.failed_photo_ids, ("p",))
        self.assertEqual(result.ineligible_photo_ids, ("control",))
        self.assertEqual(result.success_rate, 0)

    def test_no_controls_cannot_pass_and_dev_sface_cannot_count(self):
        result = self.gate((PhotoResult("p", {"SFace": ModelPhotoResults({}, {})}),))
        self.assertFalse(result.passed)
        self.assertIsNone(result.success_rate)
        self.assertEqual(result.eligible_count, 0)

    def test_development_report_keeps_sface_separate_and_per_reference(self):
        sface = ModelPhotoResults({}, {
            ("export", "one"): GalleryEvaluation("valid", {"own": .3, "other": .1}, (), .3, False),
            ("blur", "one"): INCONCLUSIVE,
        })
        item = PhotoResult("p", {"SFace": sface})
        report = development_report((item,), conditions=("export", "blur"), galleries=("one",))
        self.assertEqual((report.nonmatching_comparisons, report.inconclusive_comparisons), (1, 1))
        self.assertEqual(report.per_reference_scores["p"][("export", "one")],
                         {"own": .3, "other": .1})
        self.assertFalse(self.gate((item,)).passed)


class ScoreTests(unittest.TestCase):
    def test_tie_at_fmr_boundary_is_excluded(self):
        calibration = calibrate([
            CalibrationPair(True, .9), CalibrationPair(True, .6),
            CalibrationPair(False, .5), CalibrationPair(False, .5),
            CalibrationPair(False, .2), CalibrationPair(False, None),
        ], target_fmr=.25)
        self.assertGreater(calibration.threshold, .5)
        self.assertFalse(calibration.matches(.5))
        self.assertEqual(calibration.empirical_fmr, 0)
        self.assertEqual(calibration.valid_coverage, 5 / 6)

    def test_order_statistic_allows_one_unique_false_match(self):
        pairs = [CalibrationPair(True, .9)] + [
            CalibrationPair(False, score / 1000) for score in range(1000)
        ]
        calibration = calibrate(pairs, target_fmr=.001)
        self.assertGreater(calibration.threshold, .998)
        self.assertLess(calibration.threshold, .999)
        self.assertEqual(calibration.empirical_fmr, .001)

    def test_gallery_keeps_each_reference_and_own_source(self):
        calibration = calibrate([CalibrationPair(True, 1), CalibrationPair(False, 0)])
        probe = EmbeddingResult("valid", np.array([1., 0.]))
        refs = [Reference("own", "person", np.array([1., 0.])),
                Reference("other", "other", np.array([0., 1.]))]
        result = evaluate_gallery(probe, refs, calibration,
                                  own_identity="person", own_source_name="own",
                                  expected_reference_names=("own", "other"))
        self.assertEqual(result.status, "valid")
        self.assertEqual(set(result.per_reference), {"own", "other"})
        self.assertEqual(result.own_source_score, 1)
        self.assertTrue(result.own_identity_matched)
        partial = evaluate_gallery(probe, refs[:1], calibration,
                                   own_identity="person", own_source_name="own",
                                   expected_reference_names=("own", "other"))
        self.assertEqual(partial.status, "inconclusive")
        self.assertFalse(partial.nonmatch)

    def test_face_selection_is_unambiguous(self):
        one = np.zeros((1, 15), dtype=np.float32)
        one[0, :4] = (10, 10, 30, 30)
        self.assertIsNotNone(select_face(one)[0])
        self.assertEqual(select_face(np.vstack([one, one]))[1], "multiple_faces")
        self.assertEqual(select_face(np.vstack([one, one]), (10, 10, 40, 40))[1],
                         "ambiguous_target")
        malformed = one.copy()
        malformed[0, 4] = np.nan
        self.assertEqual(select_face(malformed)[1], "invalid_detection")

    def test_detector_resize_restores_original_box_and_landmarks(self):
        blank = np.zeros((1350, 900, 3), dtype=np.uint8)
        resized = detector_frame(blank)
        self.assertEqual(resized.shape[:2], (640, 427))
        face = np.zeros((1, 15), dtype=np.float32)
        face[0, :4] = (100, 150, 120, 180)
        face[0, 4:14] = [110, 170, 190, 170, 150, 230, 120, 290, 180, 290]
        restored = restore_face_coordinates(face, (900, 1350), (427, 640))
        self.assertAlmostEqual(restored[0, 0], 100 * 900 / 427, places=3)
        self.assertAlmostEqual(restored[0, 1], 150 * 1350 / 640, places=3)
        self.assertAlmostEqual(restored[0, 4], 110 * 900 / 427, places=3)
        self.assertAlmostEqual(restored[0, 5], 170 * 1350 / 640, places=3)
        self.assertEqual(select_face(restored)[1], None)


class ImagingTests(unittest.TestCase):
    def test_export_strips_exif_and_variants_transform_box(self):
        array = np.zeros((200, 400, 3), dtype=np.uint8)
        array[:, :, 0] = 70
        exif = Image.Exif()
        exif[274] = 1
        exif[270] = "private description"
        original = BytesIO()
        Image.fromarray(array).save(original, format="JPEG", exif=exif)
        encoded = export_jpeg(decode_image(original.getvalue()))
        with Image.open(BytesIO(encoded)) as exported:
            self.assertFalse(exported.getexif())
            self.assertNotIn("icc_profile", exported.info)
        variants = make_variants(encoded, (100, 40, 300, 160))
        self.assertEqual(set(variants), {
            "export", "jpeg85_420", "jpeg75_420", "resize960",
            "half_restore", "crop90", "blur",
        })
        self.assertEqual(variants["crop90"].target_box, (80, 30, 280, 150))
        self.assertEqual(variants["jpeg85_420"].image.shape, array.shape)
        np.testing.assert_array_equal(variants["export"].image, decode_image(encoded))

    def test_small_transformed_target_is_inconclusive(self):
        image = np.zeros((200, 2000, 3), dtype=np.uint8)
        result = make_variants(export_jpeg(image), (100, 40, 125, 80))
        self.assertEqual(result["resize960"].status, "inconclusive")
        self.assertIsNone(result["resize960"].jpeg)

    def test_exif_orientation_changes_decoded_geometry(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        image[:, :10, 0] = 255
        exif = Image.Exif()
        exif[274] = 6  # 90-degree clockwise display orientation
        payload = BytesIO()
        Image.fromarray(image).save(payload, format="JPEG", exif=exif)
        decoded = decode_image(payload.getvalue())
        self.assertEqual(decoded.shape, (30, 20, 3))


if __name__ == "__main__":
    unittest.main()
