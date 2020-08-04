import unittest

import numpy as np
from numpy import testing as npt

from . import dataset_objects as do


class TestTrialFluorescenceSubsetting(unittest.TestCase):
    def setUp(self):
        self.fluo_matrix = np.array(
            [
                # Trial 0
                [[1, 2], [3, 4], [5, 6]],  # Cell 0  # Cell 1  # Cell 2
                # Trial 1
                [[7, 8], [9, 10], [11, 12]],  # Cell 0  # Cell 1  # Cell 2
            ]
        )
        self.trial_fluorescence = do.TrialFluorescence(
            self.fluo_matrix, [0, 1], 1.0 / 30.0
        )

    def test_cell_subset_by_single_int(self):
        # Test whether fluorescence is extracted correctly
        cell_to_extract = 0
        expected_fluo = self.fluo_matrix[:, cell_to_extract, :][
            :, np.newaxis, :
        ]
        actual_fluo = self.trial_fluorescence.get_cells(cell_to_extract).data
        npt.assert_array_equal(expected_fluo, actual_fluo)

        # Test whether cell labels are subsetted correctly
        npt.assert_array_equal(
            [cell_to_extract],
            self.trial_fluorescence.get_cells(cell_to_extract).cell_vec,
        )

    def test_squeezed_cell_subset_by_single_int(self):
        # Test whether fluorescence is extracted correctly
        cell_to_extract = 0
        expected_fluo = self.fluo_matrix[:, cell_to_extract, :]
        actual_fluo = self.trial_fluorescence.get_cells(
            cell_to_extract
        ).data.squeeze()
        npt.assert_array_equal(expected_fluo, actual_fluo)

    def test_cell_subset_by_pair_of_ints(self):
        # Test whether fluorescence is extracted correctly
        expected_fluo = self.fluo_matrix[:, 0:2, :]
        actual_fluo = self.trial_fluorescence.get_cells(0, 2).data
        npt.assert_array_equal(expected_fluo, actual_fluo)

        # Test whether cell labels are subsetted correctly
        npt.assert_array_equal(
            [0, 1], self.trial_fluorescence.get_cells(0, 2).cell_vec
        )

    def test_cell_subset_by_tuple_of_ints(self):
        # Test whether fluorescence is extracted correctly
        expected_fluo = self.fluo_matrix[:, 0:2, :]
        actual_fluo = self.trial_fluorescence.get_cells((0, 2)).data
        npt.assert_array_equal(expected_fluo, actual_fluo)

        # Test whether cell labels are subsetted correctly
        npt.assert_array_equal(
            [0, 1], self.trial_fluorescence.get_cells((0, 2)).cell_vec
        )

    def test_cell_subset_by_bool_mask(self):
        mask = [True, False, True]
        expected_fluo = self.fluo_matrix[:, mask, :]
        actual_fluo = self.trial_fluorescence.get_cells(mask).data
        npt.assert_array_equal(expected_fluo, actual_fluo)

        # Test whether cell labels are subsetted correctly
        npt.assert_array_equal(
            [0, 2], self.trial_fluorescence.get_cells(mask).cell_vec
        )

    def test_trial_subset_by_single_int(self):
        # Test whether fluorescence is extracted correctly
        trial_to_extract = 0
        expected_fluo = self.fluo_matrix[trial_to_extract, :, :][
            np.newaxis, :, :
        ]
        actual_fluo = self.trial_fluorescence.get_trials(trial_to_extract).data
        npt.assert_array_equal(expected_fluo, actual_fluo)

        # Test whether cell labels are subsetted correctly
        npt.assert_array_equal(
            [trial_to_extract],
            self.trial_fluorescence.get_trials(trial_to_extract).trial_vec,
        )

    def test_trial_subset_by_bool_mask(self):
        mask = [False, True]
        expected_fluo = self.fluo_matrix[mask, :, :]
        actual_fluo = self.trial_fluorescence.get_trials(mask).data
        npt.assert_array_equal(expected_fluo, actual_fluo)

        # Test whether trial labels are subsetted correctly
        npt.assert_array_equal(
            [1], self.trial_fluorescence.get_trials(mask).trial_vec
        )


class TestTrialFluorescenceSummaryStatistics(unittest.TestCase):
    def setUp(self):
        self.fluo_matrix = np.array(
            [
                # Trial 0
                [[1, 2], [3, 4], [5, 6]],  # Cell 0  # Cell 1  # Cell 2
                # Trial 1
                [[7, 8], [9, 10], [11, 12]],  # Cell 0  # Cell 1  # Cell 2
            ]
        )
        self.trial_fluorescence = do.TrialFluorescence(
            self.fluo_matrix, [0, 1], 1.0 / 30.0
        )

    def test_trial_mean(self):
        expected = self.fluo_matrix.mean(axis=0)[np.newaxis, :, :]
        actual = self.trial_fluorescence.trial_mean().data
        npt.assert_allclose(
            actual, expected, err_msg='Trial mean not correct to within tol.'
        )

    def test_trial_std(self):
        expected = self.fluo_matrix.std(axis=0)[np.newaxis, :, :]
        actual = self.trial_fluorescence.trial_std().data
        npt.assert_allclose(
            actual, expected, err_msg='Trial std not correct to within tol.'
        )

    def test_trial_num_isnan_after_mean(self):
        tr_mean = self.trial_fluorescence.trial_mean()
        self.assertEqual(
            len(tr_mean.trial_vec),
            1,
            'Expected only 1 trial after taking mean.',
        )
        self.assertTrue(
            np.isnan(tr_mean.trial_vec[0]),
            'Expected trial_num to be NaN after taking mean across trials',
        )

    def test_trial_num_isnan_after_std(self):
        tr_mean = self.trial_fluorescence.trial_std()
        self.assertEqual(
            len(tr_mean.trial_vec),
            1,
            'Expected only 1 trial after taking std.',
        )
        self.assertTrue(
            np.isnan(tr_mean.trial_vec[0]),
            'Expected trial_num to be NaN after taking std across trials',
        )


class TestTrialFluorescenceIterators(unittest.TestCase):
    def setUp(self):
        self.fluo_matrix = np.array(
            [
                # Trial 0
                [[1, 2], [3, 4], [5, 6]],  # Cell 0  # Cell 1  # Cell 2
                # Trial 1
                [[7, 8], [9, 10], [11, 12]],  # Cell 0  # Cell 1  # Cell 2
            ]
        )
        self.trial_fluorescence = do.TrialFluorescence(
            self.fluo_matrix, [0, 1], 1.0 / 30.0
        )

    def test_trial_iterator(self):
        for trial_num, trial_data in self.trial_fluorescence.iter_trials():
            npt.assert_array_equal(
                trial_data.data,
                self.fluo_matrix[trial_num, ...][np.newaxis, :, :],
            )

    def test_cell_iterator(self):
        for cell_num, cell_data in self.trial_fluorescence.iter_cells():
            npt.assert_array_equal(
                cell_data.data,
                self.fluo_matrix[:, cell_num, :][:, np.newaxis, :],
            )


class TestTrialEyeTrackingSummaryStatistics(unittest.TestCase):
    def setUp(self):
        self.eyetracking_dict = {
            "eye_area": [[1, 2, 3], [4, 5, 6]],
            "pupil_area": [[5, 2, 4], [8, 4, 5]],
            "x_pos_deg": [[2, 4, 3], [7, 3, 4]],
            "y_pos_deg": [[7, 3, 4], [8, 4, 5]],
        }
        self.trial_eyetracking = do.TrialEyeTracking(
            self.eyetracking_dict, [0, 1], 1.0 / 30.0
        )

    def test_trial_mean(self):
        expected = {
            key: np.mean(val, axis=0)[np.newaxis, :]
            for key, val in self.eyetracking_dict.items()
        }
        actual = self.trial_eyetracking.trial_mean().data

        self.assertEqual(
            actual.keys(),
            expected.keys(),
            "Expected and actual data attributes are different:"
            " {} vs {}".format(expected.keys(), actual.keys()),
        )
        for key in expected.keys():
            npt.assert_allclose(
                actual[key],
                expected[key],
                err_msg='{} trial mean not correct to within tol.'.format(key),
            )

    def test_trial_std(self):
        expected = {
            key: np.std(val, axis=0)[np.newaxis, :]
            for key, val in self.eyetracking_dict.items()
        }
        actual = self.trial_eyetracking.trial_std().data

        self.assertEqual(
            actual.keys(),
            expected.keys(),
            "Expected and actual data attributes are different:"
            " {} vs {}".format(expected.keys(), actual.keys()),
        )
        for key in expected.keys():
            npt.assert_allclose(
                actual[key],
                expected[key],
                err_msg='{} trial std not correct to within tol.'.format(key),
            )

    def test_trial_num_isnan_after_mean(self):
        tr_mean = self.trial_eyetracking.trial_mean()
        self.assertEqual(
            len(tr_mean.trial_vec),
            1,
            'Expected only 1 trial after taking mean.',
        )
        self.assertTrue(
            np.isnan(tr_mean.trial_vec[0]),
            'Expected trial_num to be NaN after taking mean across trials',
        )

    def test_trial_num_isnan_after_std(self):
        tr_mean = self.trial_eyetracking.trial_std()
        self.assertEqual(
            len(tr_mean.trial_vec),
            1,
            'Expected only 1 trial after taking std.',
        )
        self.assertTrue(
            np.isnan(tr_mean.trial_vec[0]),
            'Expected trial_num to be NaN after taking std across trials',
        )


if __name__ == '__main__':
    unittest.main()
