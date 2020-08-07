from types import MappingProxyType

import numpy as np

from ..dataset_objects import TrialDataset, TimeseriesDataset
from ..doorman import Unlockable


class OrientationTuningExperiment(TrialDataset, TimeseriesDataset, Unlockable):
    # TODO: implement iter_cells() with shallow copies
    def __init__(
        self,
        fluorescence: TrialFluorescence,
        eyetracking: TrialEyeTracking,
        running_speed: RunningSpeed,
        trial_timetable,
    ):
        self._data = {
            "fluorescence": fluorescence,
            "eyetracking": eyetracking,
            "running_speed": running_speed,
            "trial_timetable": trial_timetable,
        }
        self.data = MappingProxyType(self._data)
        self._lock()

    @property
    def _trial_num(self):
        trial_nums = [val.trial_vec for val in self.data.values()]
        assert all([np.array_equal(trial_nums[0], tn) for tn in trial_nums])
        return trial_nums[0]

    def _lock(self):
        """Lock self to prevent accidental data modifications.

        Should mainly be called by Doorman.

        """
        for key in self.data:
            if issubclass(type(self.data[key]), Unlockable):
                self.data[key]._lock()

    def _unlock(self):
        """Unlock self to allow modifications.

        Should only ever be called by Doorman.

        """
        for key in self.data:
            if issubclass(type(self.data[key]), Unlockable):
                self.data[key]._unlock()

    def get_trials(
        self,
        *args,
        center='any',
        surround='any',
        spatial_frequency='any',
        temporal_frequency='any'
    ):
        """Get a subset of trials.

        This function can be called with positional arguments specifying a
        boolean mask or an integer range of trials to select, or with keyword
        arguments to select trials based on the visual stimulus. For help
        calling `get_trials()` with positional arguments, see
        `TrialDataset.get_trials()`.

        Keyword Parameters
        ------------------
        center : Orientation or `any`
            The orientation of the center part of the stimulus.
        surround : Orientation or {`any`, `ortho`, `iso`}
            The orientation of the surround part of the stimulus. Either a
            specific orientation, or the orientation relative to the center
            (ie, `iso` for the same orientation as center, or `ortho` for
            either orientation 90 deg to center).
        spatial_frequency : SpatialFrequency or `any`
        temporal_frequency : TemporalFrequency or `any`

        Returns
        -------
        trial_subset : OrientationTuningExperiment
            Experiment with a subset of trials matching the selection criteria.

        Raises
        ------
        SetMembershipError
            If one of the keyword arguments is not a valid value for the
            corresponding part of the stimulus. See `conditions.py` for
            details.

        See Also
        --------
        TrialDataset.get_trials()

        """
        if len(args) == 0:
            # Get trials based on stimulus parameters using keyword arguments.
            mask = np.ones(self.num_trials, dtype=np.bool_)

            for arg, attr in zip(
                [
                    (center, 'center_orientation'),
                    (surround, 'surround_orientation'),
                    (spatial_frequency, 'spatial_frequency'),
                    (temporal_frequency, 'temporal_frequency'),
                ]
            ):
                if not isinstance(arg, str):
                    # If argument is not a string, assume it can be coerced
                    # to the correct stimulus parameter type (Orientation,
                    # SpatialFrequency, etc). If it cannot, a
                    # SetMembershipError will be raised.
                    mask &= self.data['trial_timetable'][
                        'center_surround'
                    ].apply(
                        lambda stim: getattr(stim, attr) in np.atleast_1d(arg)
                    ).to_numpy()
                elif arg not in ('any', 'ortho', 'iso'):
                    raise ValueError(
                        'Invalid argument value {}, see help'.format(arg)
                    )

            if surround == 'ortho':
                mask &= self.data['trial_timetable']['center_surround'].apply(
                    lambda stim: stim.surround_is_ortho()
                ).to_numpy()
            elif surround == 'iso':
                mask &= self.data['trial_timetable']['center_surround'].apply(
                    lambda stim: stim.surround_is_iso()
                ).to_numpy()

            result = super().get_trials(mask)
        else:
            # Get trials using positional arguments only.
            # Raise an error if any keyword arguments have been set to
            # non-default values.
            kwargs = [center, surround, spatial_frequency, temporal_frequency]
            if any([arg != 'any' for arg in kwargs]):
                raise ValueError(
                    '`get_trials()` can be called with either positional '
                    'arguments or keyword arguments, not both'
                )
            del kwargs

            result = super().get_trials(*args)

        return result

    def _get_trials_from_mask(self, mask):
        return self._forward_method('get_trials', mask)

    def trial_mean(self, ignore_nan=False):
        """Get the mean across trials."""
        return self._forward_method('trial_mean', ignore_nan=ignore_nan)

    def trial_std(self, ignore_nan=False):
        """Get the standard deviation across trials."""
        return self._forward_method('trial_std', ignore_nan=ignore_nan)

    @property
    def num_timesteps(self):
        """Get the number of timesteps per trial.

        Guaranteed to be the same for all attached datasets.

        """
        num_timesteps_ = [val.num_timesteps for val in self.data.values()]
        assert all([num_timesteps_[0] == nt for nt in num_timesteps_])
        return num_timesteps_[0]

    def get_frame_range(self, start, stop=None):
        """Get a time range by frame number."""
        return self._forward_method('get_frame_range', start, stop)

    def time_mean(self, ignore_nan=False):
        """Get the mean over time within each trial."""
        return self._forward_method('time_mean', ignore_nan=ignore_nan)

    def time_std(self, ignore_nan=False):
        """Get the standard deviation over time within each trial."""
        return self._forward_method('time_std', ignore_nan=ignore_nan)

    def _forward_method(self, method_name, *args, **kwargs):
        result = self.copy()

        new_data = {}
        for key in self.data:
            if callable(getattr(self.data[key], method_name, None)):
                # If the class has `method_name`, use it to subset data
                new_data[key] = getattr(self.data[key], method_name)(
                    *args, **kwargs
                ).data
            elif hasattr(self.data, 'data'):
                # If the class doesn't have `method_name`, skip subsetting data
                new_data[key] = self.data[key].data
            else:
                # Fallback to just getting the data attribute.
                new_data[key] = self.data[key]

        result._data = new_data

        return result

    def plot(self, **pltargs):
        raise NotImplementedError
        if self.data["fluorescence"].num_cells == 1:
            # TODO: use code from analysis/orientation_plots.py
            pass
        else:
            # Method to make a summary plot for all cells from one session
            NotImplemented

    def apply_quality_control(self):
        raise NotImplementedError
