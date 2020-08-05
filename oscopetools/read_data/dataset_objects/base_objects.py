"""Abstract base classes for dataset-like objects.

If you aren't sure how to use the `Dataset` objects produced by
`read_data.get_eye_tracking()`, `read_data.get_dff_traces()`, etc., read this
first. `Dataset`, `TrialDataset`, and `TimeseriesDataset` define most
of the functionality of dataset-like objects. Together, they provide a suite
of methods for slicing (`get_time_range()`, `get_trials()`, etc.), summarizing
(`trial_mean()`, `time_std()`), and visualizing (`plot()`) data.

There are three main reasons to use `Dataset` objects in your code:

1. They provide a lot of useful functionality (see examples above).
2. They will make your code easier to understand
   (`fluorescence[mask, :, :].mean(axis=0)` vs.
   `fluorescence.get_trials(mask).trial_mean()`).
3. They reduce bugs by preventing accidental modifications of sensitive data
   and carefully handling special cases so you don't have to.

If you aren't convinced by this sales pitch or for cases when only an ndarray
will do the job, you can always get a more primitive view of a `Dataset`
through its `data` attribute.

"""
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from .util import (
    try_parse_positionals_as_slice_like,
    get_vector_mask_from_range,
    validate_vector_mask_length,
    SliceParseError
)

class Dataset(ABC):
    """A dataset that is interesting to analyze on its own."""

    @abstractmethod
    def __init__(self):
        self._clean = False  # Whether quality control has been applied

    @abstractmethod
    def plot(self, ax=None, **pltargs):
        """Display a diagnostic plot.

        Parameters
        ----------
        ax : matplotlib.Axes object or None
            Axes object onto which to draw the diagnostic plot. Defaults to the
            current Axes if None.
        pltargs
            Parameters passed to `plt.plot()` (or similar) as keyword
            arguments. See `plt.plot` for a list of valid arguments. Examples:
            `color='red'`, `linestyle='dashed'`.

        Returns
        -------
        axes : Axes
            Axes object containing the diagnostic plot.

        """
        # Suggested implementation for derived classes:
        # def plot(self, type_specific_arguments, ax=None, **pltargs):
        #     ax = super().plot(ax=ax, **pltargs)
        #     ax.plot(relevant_data, **pltargs)  # pltargs might include color, linestyle, etc
        #     return ax  # ax should be returned so the user can change axis labels, etc

        # Default to the current Axes if none are supplied.
        if ax is None:
            ax = plt.gca()

        return ax

    @abstractmethod
    def apply_quality_control(self, inplace=False):
        """Clean up the dataset.

        Parameters
        ----------
        inplace : bool, default False
            Whether to clean up the current Dataset instance (ie, self) or
            a copy. In either case, a cleaned Dataset instance is returned.

        Returns
        -------
        dataset : Dataset
            A cleaned dataset.

        """
        # Suggested implementation for derived classes:
        # def apply_quality_control(self, type_specific_arguments, inplace=False):
        #     dset_to_clean = super().apply_quality_control(inplace)
        #     # Do stuff to `dset_to_clean` to clean it.
        #     dset_to_clean._clean = True
        #     return dset_to_clean

        # Get a reference to the dataset to be cleaned. Might be the current
        # dataset or a copy of it.
        if inplace:
            dset_to_clean = self
        else:
            dset_to_clean = self.copy()

        return dset_to_clean

    def copy(self):
        """Get a deep copy of the current Dataset."""
        return deepcopy(self)


class TimeseriesDataset(Dataset):
    """Abstract base class for Datasets containing timeseries."""

    def __init__(self, timestep_width):
        self._timestep_width = timestep_width

    def __len__(self):
        return self.num_timesteps

    @property
    @abstractmethod
    def num_timesteps(self):
        """Number of timesteps in timeseries."""
        raise NotImplementedError

    @property
    def timestep_width(self):
        """Width of each timestep in seconds."""
        return self._timestep_width

    @property
    def duration(self):
        """Duration of the timeseries in seconds."""
        return self.num_timesteps * self.timestep_width

    @property
    def time_vec(self):
        """A vector of timestamps the same length as the timeseries."""
        time_vec = np.arange(
            0, self.duration - 0.5 * self.timestep_width, self.timestep_width
        )
        assert len(time_vec) == len(
            self
        ), "Length of time_vec ({}) does not match instance length ({})".format(
            len(time_vec), len(self)
        )
        return time_vec

    def get_time_range(self, start, stop=None):
        """Extract a time window from the timeseries by time in seconds.

        Parameters
        ----------
        start, stop : float
            Beginning and end of the time window to extract in seconds. If
            `stop` is omitted, only the frame closest to `start` is returned.

        Returns
        -------
        windowed_timeseries : TimeseriesDataset
            A timeseries of the same type as the current instance containing
            only the frames in the specified window. Note that the `time_vec`
            of `windowed_timeseries` will start at 0, not `start`.

        """
        frame_range = [
            self._get_nearest_frame(t_)
            for t_ in (start, stop)
            if t_ is not None
        ]
        return self.get_frame_range(*frame_range)

    @abstractmethod
    def get_frame_range(self, start, stop=None):
        """Extract a time window from the timeseries by frame number.

        Parameters
        ----------
        start, stop : int
            Beginning and end of the time window to extract in frames. If
            `stop` is omitted, only the `start` frame is returned.

        Returns
        -------
        windowed_timeseries : TimeseriesDataset
            A timeseries of the same type as the current instance containing
            only the frames in the specified window. Note that the `time_vec`
            of `windowed_timeseries` will start at 0, not `start`.

        """
        raise NotImplementedError

    def _get_nearest_frame(self, time_):
        """Round a timestamp to the nearest integer frame number."""
        frame_num = np.argmin(np.abs(self.time_vec - time_))
        assert frame_num <= len(self)

        return min(frame_num, len(self) - 1)

    @abstractmethod
    def time_mean(self, ignore_nan=False):
        """Get the mean of the timeseries over time."""
        raise NotImplementedError

    @abstractmethod
    def time_std(self, ignore_nan=False):
        """Get the std of the timeseries over time."""
        raise NotImplementedError


class TrialDataset(Dataset):
    """Abstract base class for datasets that are divided into trials.

    All children should have a list-like `_trial_num` attribute.

    """

    @property
    def num_trials(self):
        """Number of trials."""
        return len(self._trial_num)

    @property
    def trial_vec(self):
        """Trial numbers."""
        return self._trial_num

    def get_trials(self, *args):
        """Get a subset of the trials in TrialDataset.

        Parameters
        ----------
        start, stop : int
            Get a range of trials from `start` to an optional `stop`.
        mask : bool vector-like
            A boolean mask used to select trials.

        Returns
        -------
        trial_subset : TrialDataset
            A new `TrialDataset` object containing only the specified trials.

        """
        # Implementation note:
        # This function tries to convert positional arguments to a boolean
        # trial mask. `_get_trials_from_mask` is reponsible for actually
        # getting the `trial_subset` to be returned.
        try:
            # Try to parse positional arguments as a range of trials
            trial_range = try_parse_positionals_as_slice_like(*args)
            mask = get_vector_mask_from_range(self.trial_vec, *trial_range)
        except SliceParseError:
            # Couldn't parse pos args as a range of trials. Try parsing as
            # a boolean trial mask.
            if len(args) == 1:
                mask = validate_vector_mask_length(args[0], self.num_trials)
            else:
                raise ValueError(
                    "Expected a single mask argument, got {}".format(len(args))
                )

        return self._get_trials_from_mask(mask)

    def iter_trials(self):
        """Get an iterator over all trials.

        Yields
        ------
        (trial_num, trial_contents): (int, TrialDataset)
            Yields a tuple containing the trial number and a `TrialDataset`
            containing that trial for each trial in the original
            `TrialDataset`.

        Example
        -------
        >>> trials = TrialDataset()
        >>> for trial_num, trial in trials.iter_trials():
        >>>     print(trial_num)
        >>>     trial.plot()

        """
        for trial_num in self.trial_vec:
            yield (trial_num, self.get_trials(trial_num))

    @abstractmethod
    def _get_trials_from_mask(self, mask):
        """Get a subset of trials using a boolean mask.

        Subclasses are required to implement this method to get the rest of
        TrialDataset functionality.

        Parameters
        ----------
        mask : bool vector-like
            A boolean trial mask, the length of which is guaranteed to match
            the number of trials.

        Returns
        -------
        trial_subset : TrialDataset
            A new `TrialDataset` object containing only the specified trials.

        """
        raise NotImplementedError

    @abstractmethod
    def trial_mean(self):
        """Get the mean across all trials.

        Returns
        -------
        trial_mean : TrialDataset
            A new `TrialDataset` object containing the mean of all trials in
            the current one.

        """
        raise NotImplementedError

    @abstractmethod
    def trial_std(self):
        """Get the standard deviation across all trials.

        Returns
        -------
        trial_std : TrialDataset
            A new `TrialDataset` object containing the standard deviation of
            all trials in the current one.

        """
        raise NotImplementedError


