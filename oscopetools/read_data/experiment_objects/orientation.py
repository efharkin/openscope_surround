from types import MappingProxyType

import numpy as np

from ..dataset_objects import TrialDataset, TimeseriesDataset
from ..doorman import Unlockable


class OrientationTuningExperiment(TrialDataset, TimeseriesDataset, Unlockable):
    # TODO: implement iter_cells() with shallow copies
    # TODO: implement method to select trials by stimulus parameters
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
        assert all([num_timesteps_[0] = nt for nt in num_timesteps_])
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
            new_data[key] = getattr(self.data[key], method_name)(
                *args, **kwargs
            )

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
error: cannot format -: Cannot parse: 73:38:         assert all([num_timesteps_[0] = nt for nt in num_timesteps_])
