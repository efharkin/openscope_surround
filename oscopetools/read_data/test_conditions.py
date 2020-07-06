import unittest

import numpy as np

from . import conditions as cond


class OrientationOrdering(unittest.TestCase):
    def test_le_numeric(self):
        self.assertLess(cond.Orientation(45), cond.Orientation(90))
        self.assertLess(cond.Orientation(90), cond.Orientation(180))

    def test_le_none(self):
        self.assertLess(cond.Orientation(None), cond.Orientation(45))
        self.assertLess(cond.Orientation(None), cond.Orientation(0))

    def test_le_nan(self):
        self.assertLess(cond.Orientation(np.nan), cond.Orientation(45))
        self.assertLess(cond.Orientation(np.nan), cond.Orientation(0))

    def test_ge_numeric(self):
        self.assertGreater(cond.Orientation(90), cond.Orientation(45))

    def test_ge_none(self):
        self.assertGreater(cond.Orientation(45), cond.Orientation(None))
        self.assertGreater(cond.Orientation(0), cond.Orientation(None))

    def test_ge_nan(self):
        self.assertGreater(cond.Orientation(45), cond.Orientation(np.nan))
        self.assertGreater(cond.Orientation(0), cond.Orientation(np.nan))

    def test_eq_numeric(self):
        self.assertEqual(cond.Orientation(45), cond.Orientation(45))
        self.assertEqual(cond.Orientation(90), cond.Orientation(90.0))

    def test_eq_none(self):
        """Assert that None orientation is equal to itself."""
        self.assertEqual(cond.Orientation(None), cond.Orientation(None))
        self.assertNotEqual(cond.Orientation(None), cond.Orientation(0))

    def test_eq_nan(self):
        """Assert that NaN orientation is equal to itself."""
        self.assertEqual(cond.Orientation(np.nan), cond.Orientation(np.nan))
        self.assertNotEqual(cond.Orientation(np.nan), cond.Orientation(0))

    def test_eq_nan_none(self):
        """Assert that None and NaN orientations are equal."""
        self.assertEqual(cond.Orientation(np.nan), cond.Orientation(None))


class OrientationIteration(unittest.TestCase):
    def test_iteration(self):
        """Iteration yield all allowed values + None"""
        for allowed_value, orientation_value in zip(
            cond.Orientation._MEMBERS, cond.Orientation
        ):
            self.assertEqual(orientation_value, allowed_value)


class CenterSurroundStimulusEquality(unittest.TestCase):
    def test_equal_all_numeric(self):
        css1 = cond.CenterSurroundStimulus(2.0, 0.04, 0.8, 45, 90)
        css2 = cond.CenterSurroundStimulus(2.0, 0.04, 0.8, 45, 90)
        self.assertEqual(css1, css2)

    def test_neq_all_numeric(self):
        css1 = cond.CenterSurroundStimulus(2.0, 0.04, 0.8, 45, 90)

        # Not equal if surround orientation differs
        css2 = cond.CenterSurroundStimulus(2.0, 0.04, 0.8, 45, 45)
        self.assertNotEqual(css1, css2)

        # Not equal if center orientation differs
        css2 = cond.CenterSurroundStimulus(2.0, 0.04, 0.8, 90, 90)
        self.assertNotEqual(css1, css2)

        # Not equal if temporal_frequency differs
        css2 = cond.CenterSurroundStimulus(1.0, 0.04, 0.8, 45, 90)
        self.assertNotEqual(css1, css2)

    def test_eq_all_none(self):
        css1 = cond.CenterSurroundStimulus(1.0, 0.04, 0.8, None, None)
        css2 = cond.CenterSurroundStimulus(1.0, 0.04, 0.8, None, None)
        self.assertEqual(css1, css2)

    def test_eq_all_nan(self):
        css1 = cond.CenterSurroundStimulus(1.0, 0.04, 0.8, np.nan, np.nan)
        css2 = cond.CenterSurroundStimulus(1.0, 0.04, 0.8, np.nan, np.nan)
        self.assertEqual(css1, css2)

    def test_eq_nan_none(self):
        css1 = cond.CenterSurroundStimulus(1.0, 0.04, 0.8, np.nan, np.nan)
        css2 = cond.CenterSurroundStimulus(1.0, 0.04, 0.8, None, None)
        self.assertEqual(css1, css2)

    def test_neq_some_none(self):
        css1 = cond.CenterSurroundStimulus(2.0, 0.04, 0.8, 45, 90)

        # Not equal if surround orientation differs
        css2 = cond.CenterSurroundStimulus(2.0, 0.04, 0.8, 45, None)
        self.assertNotEqual(css1, css2)

        # Not equal if center orientation differs
        css2 = cond.CenterSurroundStimulus(2.0, 0.04, 0.8, None, 90)
        self.assertNotEqual(css1, css2)
