import unittest
from fckface_lab.datasets import identity_split


class SplitTests(unittest.TestCase):
    def test_splits_are_identity_disjoint_and_order_independent(self):
        ids = [f"frll-{number:03}" for number in range(102)]
        actual = identity_split(ids)
        self.assertEqual(actual, identity_split(list(reversed(ids)) + ids))
        self.assertEqual([sum(v == split for v in actual.values()) for split in
                          ("development", "calibration", "held_out")], [60, 20, 22])

    def test_incomplete_identity_inventory_does_not_silently_resplit(self):
        with self.assertRaises(ValueError):
            identity_split(["frll-001"])


if __name__ == "__main__":
    unittest.main()
