"""Small deterministic checks for the research-only appearance renderers."""

import unittest

import numpy as np

from fckface_lab.patterns import (METHOD_DEFAULTS, actual_distortion, face_mask,
                                  render, scale_to_rms)


def sample():
    yy, xx = np.mgrid[:150, :170]
    texture = ((xx // 4 + yy // 4) % 2) * 5
    image = np.stack((40 + xx * .6 + yy * .2 + texture,
                      60 + xx * .4 + yy * .3 + texture,
                      75 + xx * .3 + yy * .4 + texture), axis=2).astype(np.uint8)
    box = (43, 23, 82, 105)
    landmarks = np.array([[69, 61], [98, 62], [84, 83], [73, 104], [96, 104]],
                         dtype=np.float32)
    return image, box, landmarks


class PatternTests(unittest.TestCase):
    def test_all_methods_preserve_source_and_outside_face(self):
        image, box, landmarks = sample()
        source = image.copy()
        outside = face_mask(image.shape, box) == 0
        for method in METHOD_DEFAULTS:
            with self.subTest(method=method):
                edited = render(image, box, landmarks, method, seed=42)
                self.assertEqual(edited.dtype, np.uint8)
                self.assertEqual(edited.shape, image.shape)
                self.assertFalse(np.shares_memory(edited, image))
                self.assertTrue(np.array_equal(edited[outside], image[outside]))
                metrics = actual_distortion(image, edited, box, landmarks)
                self.assertEqual(metrics["changed_outside_face_pixels"], 0)
                if method != "unedited":
                    self.assertGreater(metrics["changed_face_pixels"], 0)
        self.assertTrue(np.array_equal(image, source))

    def test_seed_and_yunet_row_replay(self):
        image, box, landmarks = sample()
        row = np.array([*box, *landmarks.flatten(), .99])
        first = render(image, box, landmarks, "face_relative_multiscale", seed=11)
        second = render(image, row, None, "face_relative_multiscale", seed=11)
        third = render(image, box, landmarks, "face_relative_multiscale", seed=12)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, third))

    def test_face_relative_translation_and_scale(self):
        image, box, landmarks = sample()
        dx, dy = 11, 13
        moved = np.roll(image, (dy, dx), axis=(0, 1))
        moved_box = (box[0] + dx, box[1] + dy, box[2], box[3])
        moved_landmarks = landmarks + np.array([dx, dy])
        relative = render(image, box, landmarks, "face_relative_multiscale", seed=7)
        shifted_relative = render(moved, moved_box, moved_landmarks,
                                  "face_relative_multiscale", seed=7)
        self.assertTrue(np.array_equal(relative,
                         np.roll(shifted_relative, (-dy, -dx), axis=(0, 1))))
        global_pattern = render(image, box, landmarks, "global_multiscale", seed=7)
        shifted_global = render(moved, moved_box, moved_landmarks,
                                "global_multiscale", seed=7)
        self.assertFalse(np.array_equal(global_pattern,
                          np.roll(shifted_global, (-dy, -dx), axis=(0, 1))))

        # The same face doubled in resolution retains its normalized carrier
        # placement; raster edges and uint8 rounding need not be identical.
        doubled = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
        larger = render(doubled, tuple(2 * v for v in box), landmarks * 2,
                        "face_relative_multiscale", seed=7)
        self.assertLess(float(np.mean(np.abs(larger[::2, ::2].astype(int) -
                                           relative.astype(int)))), 2.0)

    def test_rms_matching_or_explicit_mismatch(self):
        image, box, landmarks = sample()
        proposal = render(image, box, landmarks, "graphic_halftone")
        ordinary = render(image, box, landmarks, "ordinary_halftone",
                          {"step": 36})
        self.assertFalse(np.array_equal(proposal, ordinary))
        available = actual_distortion(image, proposal, box)["face_rms"]
        self.assertGreater(available, 0)
        smaller, measured = scale_to_rms(image, proposal, box, available / 2)
        self.assertTrue(measured["rms_matched"])
        self.assertLess(measured["face_rms"], available)
        self.assertEqual(measured["changed_outside_face_pixels"], 0)
        unchanged, missed = scale_to_rms(image, proposal, box, available + 10)
        self.assertTrue(np.array_equal(unchanged, proposal))
        self.assertFalse(missed["rms_matched"])
        self.assertLess(missed["rms_mismatch"], 0)
        zero, zero_metrics = scale_to_rms(image, proposal, box, 0)
        self.assertTrue(np.array_equal(zero, image))
        self.assertEqual(zero_metrics["face_rms"], 0)
        self.assertTrue(zero_metrics["rms_matched"])


if __name__ == "__main__":
    unittest.main()
