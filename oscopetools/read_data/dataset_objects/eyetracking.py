"""Objects for interacting with eyetracking datasets."""

__all__ = ("RawEyeTracking", "TrialEyeTracking")

from copy import deepcopy
import warnings
from types import MappingProxyType

import numpy as np
import seaborn as sns

from ..doorman import Unlockable
from .base_objects import TimeseriesDataset, TrialDataset
from .util import robust_range, ROBUST_PLOT_RANGE_DEFAULT_HALF_WIDTH


class EyeTracking(TimeseriesDataset, Unlockable):
    _DATA_MEMBER_NAMES = ["eye_area", "pupil_area", "x_pos_deg", "y_pos_deg"]

    def __init__(self, tracked_attributes: dict, timestep_width: float):
        super().__init__(timestep_width)

        self._data = {}
        for key in self._DATA_MEMBER_NAMES:
            self._data[key] = np.asarray(tracked_attributes[key])

        # Add `self.data` as a read-only interface to `_data`
        self.data = MappingProxyType(self._data)
        self._lock()

        # Check that `_data` has been initialized properly and raise error if not
        try:
            self._assert_data_shapes_equal()
        except RuntimeError:
            raise ValueError(
                'Expected attributes in `tracked_attributes` to all have the same shape'
            )

    def _unlock(self):
        """Temporarily unlock self.

        Should only ever be called by Doorman.

        """
        for key in self.data:
            self.data[key].flags.writeable = True

    def _lock(self):
        """Lock self to prevent accidental modification.

        Should mainly be called by Doorman.

        """
        for key in self.data:
            self.data[key].flags.writeable = False

    def __copy__(self):
        """Return a shallow copy of self."""
        cls = self.__class__
        new_obj = cls.__new__(cls)

        for key, val in self.__dict__.items():
            if key != "data":
                setattr(new_obj, key, val)

        new_obj.data = MappingProxyType(new_obj._data)

        return new_obj

    def __deepcopy__(self, memo):
        """Return a deep copy of self."""
        cls = self.__class__
        new_obj = cls.__new__(cls)

        memo[id(self)] = new_obj

        for key, val in self.__dict__.items():
            if key != "data":
                setattr(new_obj, key, deepcopy(val, memo))

        new_obj.data = MappingProxyType(new_obj._data)

        return new_obj

    def _assert_data_shapes_equal(self):
        """Raise an error if not all attrs of `data` have the same length."""
        shapes = [np.shape(data_item) for data_item in self.data.values()]
        if not all([shape_ == shapes[0] for shape_ in shapes]):
            raise RuntimeError('Not all data attributes have same length')

    def copy(self, read_only=False):
        """Get a deep copy.

        Parameters
        ----------
        read_only : bool, default False
            Whether to get a read-only copy of the underlying `data` arrays.
            Getting a read-only copy is much faster and should be used if a
            large number of copies need to be created.

        """
        if read_only:
            # Get a read-only view of the data arrays
            # This is much faster than creating a full copy
            read_only_data = {}
            for key in self.data:
                read_only_data[key] = self.data[key].view()
                read_only_data[key].flags.writeable = False

            deepcopy_memo = {id(self._data): read_only_data}
            copy_ = deepcopy(self, deepcopy_memo)
        else:
            copy_ = deepcopy(self)

        return copy_

    @property
    def num_timesteps(self):
        """Number of timesteps in EyeTracking dataset."""
        self._assert_data_shapes_equal()
        return self.data["eye_area"].shape[-1]

    def get_frame_range(self, start: int, stop: int = None):
        window = self.copy(read_only=True)

        for key, val in window.data.values():
            if stop is None:
                window._data[key] = np.atleast_1d(val[..., start])
            else:
                window._data[key] = np.atleast_1d(val[..., start:stop])

        window._assert_data_shapes_equal()

        return window

    def time_mean(self, ignore_nan=False):
        """Mean of the eye tracking data for each time period."""
        eyetracking_copy = self.copy(read_only=True)

        if ignore_nan:
            eyetracking_copy._data = np.nanmean(
                eyetracking_copy.data, axis=eyetracking_copy.data.ndim - 1
            )[..., np.newaxis]
        else:
            eyetracking_copy._data = np.mean(
                eyetracking_copy.data, axis=eyetracking_copy.data.ndim - 1
            )[..., np.newaxis]

        return eyetracking_copy

    def time_std(self, ignore_nan=False):
        """Standard deviation of the eye tracking data for each time period."""
        eyetracking_copy = self.copy(read_only=True)

        if ignore_nan:
            eyetracking_copy._data = np.nanstd(
                eyetracking_copy.data, axis=eyetracking_copy.data.ndim - 1
            )[..., np.newaxis]
        else:
            eyetracking_copy._data = np.std(
                eyetracking_copy.data, axis=eyetracking_copy.data.ndim - 1
            )[..., np.newaxis]

        return eyetracking_copy


class RawEyeTracking(EyeTracking):
    def __init__(self, tracked_attributes: dict, timestep_width: float):
        super().__init__(tracked_attributes, timestep_width)

    def cut_by_trials(
        self,
        trial_timetable,
        num_baseline_frames=None,
        both_ends_baseline=False,
    ):
        """Divide eye tracking parameters up into equal-length trials.

        Parameters
        ----------
        trial_timetable : pd.DataFrame-like
            A DataFrame-like object with 'Start' and 'End' items for the start
            and end frames of each trial, respectively.

        Returns
        -------
        trial_eyetracking : TrialEyeTracking

        """
        if ("Start" not in trial_timetable) or ("End" not in trial_timetable):
            raise ValueError(
                "Could not find `Start` and `End` in trial_timetable."
            )

        if (num_baseline_frames is None) or (num_baseline_frames < 0):
            num_baseline_frames = 0

        trial_data = {key: [] for key in self._DATA_MEMBER_NAMES}

        num_frames = []
        for start, end in zip(
            trial_timetable["Start"], trial_timetable["End"]
        ):
            # Coerce `start` and `end` to ints if possible
            if (int(start) != start) or (int(end) != end):
                raise ValueError(
                    "Expected trial start and end frame numbers"
                    " to be ints, got {} and {} instead".format(start, end)
                )
            start = max(int(start) - num_baseline_frames, 0)

            # Optionally, pad the end of the trial with a post-trial baseline
            if both_ends_baseline:
                end = int(end) + num_baseline_frames
            else:
                end = int(end)

            # Append data from this trial
            for key in trial_data:
                trial_data[key].append(self.data[start:end])

            num_frames.append(end - start)

        # Truncate all trials to the same length if necessary
        min_num_frames = min(num_frames)
        if not all([dur == min_num_frames for dur in num_frames]):
            warnings.warn(
                "Truncating all trials to shortest duration {} "
                "frames (longest trial is {} frames)".format(
                    min_num_frames, max(num_frames)
                )
            )
            for key in trial_data:
                trial_data[key] = [
                    tr[:min_num_frames] for tr in trial_data[key]
                ]

        # Try to get a vector of trial numbers
        try:
            trial_num = trial_timetable["trial_num"]
        except KeyError:
            try:
                trial_num = trial_timetable.index.tolist()
            except AttributeError:
                warnings.warn(
                    "Could not get trial_num from trial_timetable. "
                    "Falling back to arange."
                )
                trial_num = np.arange(0, len(num_frames))

        # Construct TrialEyeTracking and return it.
        trial_eyetracking = TrialEyeTracking(
            trial_data, trial_num, self.timestep_width,
        )
        trial_eyetracking._baseline_duration = (
            num_baseline_frames * self.timestep_width
        )
        trial_eyetracking._both_ends_baseline = both_ends_baseline

        # Check that trial_eyetracking was constructed correctly.
        assert trial_eyetracking.num_timesteps == min_num_frames
        assert trial_eyetracking.num_trials == len(num_frames)

        return trial_eyetracking

    def plot(
        self, channel="position", robust_range_=False, ax=None, **pltargs
    ):
        """Make a diagnostic plot of eyetracking data."""
        ax = super().plot(ax, **pltargs)

        # Check whether the `channel` argument is valid
        if channel not in self.data and channel != "position":
            raise ValueError(
                "Got unrecognized channel `{}`, expected one of "
                "{} or `position`".format(channel, self.data.keys())
            )

        if channel in self.data:
            if robust_range_:
                ax.axhspan(
                    *robust_range(
                        self.data[channel],
                        half_width=1.5,
                        center="median",
                        spread="iqr",
                    ),
                    color="gray",
                    label="Median $\pm$ 1.5 IQR",
                    alpha=0.5,
                )
                ax.legend()

            ax.plot(self.time_vec, self.data[channel], **pltargs)
            ax.set_xlabel("Time (s)")

            if robust_range_:
                ax.set_ylim(
                    robust_range(
                        self.data[channel],
                        half_width=ROBUST_PLOT_RANGE_DEFAULT_HALF_WIDTH,
                    )
                )

        elif channel == "position":
            if pltargs.pop("style", None) in ["contour", "density"]:
                x = self.data["x_pos_deg"]
                y = self.data["y_pos_deg"]
                mask = np.isnan(x) | np.isnan(y)
                if any(mask):
                    warnings.warn(
                        "Dropping {} NaN entries in order to estimate "
                        "density.".format(sum(mask))
                    )
                sns.kdeplot(x[~mask], y[~mask], ax=ax, **pltargs)
            else:
                ax.plot(
                    self.data["x_pos_deg"], self.data["y_pos_deg"], **pltargs,
                )

            if robust_range_:
                # Set limits based on approx. data range, excluding outliers
                ax.set_ylim(
                    robust_range(
                        self.data["y_pos_deg"],
                        half_width=ROBUST_PLOT_RANGE_DEFAULT_HALF_WIDTH,
                    )
                )
                ax.set_xlim(
                    robust_range(
                        self.data["x_pos_deg"],
                        half_width=ROBUST_PLOT_RANGE_DEFAULT_HALF_WIDTH,
                    )
                )
            else:
                # Set limits to a 180 deg standard range
                ax.set_xlim(-90.0, 90.0)
                ax.set_ylim(-90.0, 90.0)

        else:
            raise NotImplementedError(
                "Plotting for channel {} is not implemented.".format(channel)
            )

        return ax

    def apply_quality_control(self, inplace=False):
        super().apply_quality_control(inplace)
        raise NotImplementedError


class TrialEyeTracking(EyeTracking, TrialDataset):
    """EyeTracking timeseries divided into trials."""

    def __init__(self, tracked_attributes: dict, trial_num, timestep_width):
        for key in self._DATA_MEMBER_NAMES:
            assert np.ndim(tracked_attributes[key]) == 2
            assert np.shape(tracked_attributes[key])[0] == len(trial_num)

        super().__init__(tracked_attributes, timestep_width)

        self._baseline_duration = 0
        self._both_ends_baseline = False
        self._trial_num = np.asarray(trial_num)

    def _get_trials_from_mask(self, mask):
        trial_subset = self.copy(read_only=True)

        trial_subset._trial_num = trial_subset._trial_num[mask].copy()
        for key in self.data.keys():
            trial_subset._data[key] = np.atleast_2d(
                trial_subset._data[key][mask, :].copy()
            )

        return trial_subset

    def trial_mean(self, ignore_nan=False):
        """Get the mean eye parameters across trials.

        Parameters
        ----------
        ignore_nan : bool, default False
            Whether to return the `mean` or `nanmean`.

        Returns
        -------
        trial_mean : TrialEyeTracking
            A new `TrialEyeTracking` object with the mean across trials.

        See Also
        --------
        `trial_std()`
        `time_std()`
        `time_mean()`

        """
        trial_mean = self.copy(read_only=True)
        trial_mean._trial_num = np.asarray([np.nan])

        if ignore_nan:
            for key in trial_mean._data:
                trial_mean._data[key] = trial_mean._data[key].mean(axis=0)[
                    np.newaxis, :
                ]
        else:
            for key in trial_mean._data:
                trial_mean._data[key] = np.nanmean(
                    trial_mean._data[key], axis=0
                )[np.newaxis, :]

        return trial_mean

    def trial_std(self, ignore_nan=False):
        """Get the standard deviation of the eye parameters across trials.

        Parameters
        ----------
        ignore_nan : bool, default False
            Whether to return the `std` or `nanstd`.

        Returns
        -------
        trial_std : TrialEyeTracking
            A new `TrialEyeTracking` object with the standard deviation
            across trials.

        See Also
        --------
        `trial_mean()`
        `time_mean()`
        `time_std()`

        """
        trial_std = self.copy(read_only=True)
        trial_std._trial_num = np.asarray([np.nan])

        if ignore_nan:
            for key in trial_std._data:
                trial_std._data[key] = trial_std._data[key].std(axis=0)[
                    np.newaxis, :
                ]
        else:
            for key in trial_std._data:
                trial_std._data[key] = np.nanstd(trial_std._data[key], axis=0)[
                    np.newaxis, :
                ]

        return trial_std

    def plot(self, ax=None, **pltargs):
        raise NotImplementedError

    def apply_quality_control(self):
        raise NotImplementedError
