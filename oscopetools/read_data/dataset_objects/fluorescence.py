"""Objects for interacting with fluorescence datasets."""

__all__ = ("RawFluorescence", "TrialFluorescence")

from copy import deepcopy
import warnings

import numpy as np
import matplotlib.pyplot as plt

from ..doorman import Unlockable, LockError
from .base_objects import TimeseriesDataset, TrialDataset
from .util import (
    try_parse_positionals_as_slice_like,
    get_vector_mask_from_range,
    validate_vector_mask_length,
    SliceParseError,
)


class Fluorescence(TimeseriesDataset, Unlockable):
    """A fluorescence timeseries.

    Any fluorescence timeseries. May have one or more cells and one or more
    trials.

    """

    def __init__(self, fluorescence_array, timestep_width):
        super().__init__(timestep_width)

        self._data = np.asarray(fluorescence_array)
        self._cell_vec = np.arange(0, self.num_cells)
        self.__locked = False
        self._is_z_score = False
        self._is_dff = False
        self._is_positive_clipped = False

        self._lock()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if self.__locked:
            raise LockError(
                '{} instance is currently locked to prevent '
                'accidental modification '
                '(see `Unlockable` and `unlock()` for details)'.format(
                    type(self)
                )
            )
        else:
            self._data = value

    @property
    def is_z_score(self):
        """True if the units of the data signal are in standard deviations."""
        return self._is_z_score

    @property
    def is_dff(self):
        return self._is_dff

    @property
    def is_positive_clipped(self):
        return self._is_positive_clipped

    @property
    def cell_vec(self):
        return self._cell_vec

    def _unlock(self):
        """Unlock Fluorescence object to allow modification.

        Should only ever be called by Doorman.

        """
        self._data.flags.writeable = True
        self.__locked = False

    def _lock(self):
        """Lock Fluorescence object to prevent accidental modification.

        Should mainly be used by Doorman.

        """
        self._data.flags.writeable = False
        self.__locked = True

    @property
    def num_timesteps(self):
        """Number of timesteps."""
        return self.data.shape[-1]

    @property
    def num_cells(self):
        """Number of ROIs."""
        return self.data.shape[-2]

    def get_cells(self, *args):
        # Implementation note:
        # This function tries to convert positional arguments to a boolean
        # cell mask. `_get_cells_from_mask` is reponsible for actually
        # getting the `cell_subset` to be returned.
        try:
            # Try to parse positional arguments as a range of cells
            cell_range = try_parse_positionals_as_slice_like(*args)
            mask = get_vector_mask_from_range(self.cell_vec, *cell_range)
        except SliceParseError:
            # Couldn't parse pos args as a range of cells. Try parsing as
            # a boolean cell mask.
            if len(args) == 1:
                mask = validate_vector_mask_length(args[0], self.num_cells)
            else:
                raise ValueError(
                    "Expected a single mask argument, got {}".format(len(args))
                )

        return self._get_cells_from_mask(mask)

    def iter_cells(self):
        """Get an iterator over all cells in the fluorescence dataset.

        Yields
        ------
        (cell_num, cell_fluorescence) : (int, Fluorescence)
            Yields a tuple of the cell number and fluorescence for each cell.

        Example
        -------
        >>> fluo_dset = Fluorescence()
        >>> for cell_num, cell_fluorescence in fluo_dset.iter_cells():
        >>>     print('Cell number {}'.format(cell_num))
        >>>     cell_fluorescence.plot()

        """
        for cell_num in self.cell_vec:
            yield (cell_num, self.get_cells(cell_num))

    def get_frame_range(self, start, stop=None):
        """Get a time window by frame number."""
        fluo_copy = self.copy(read_only=True)

        if stop is None:
            time_slice = self.data[..., start][..., np.newaxis]
        else:
            time_slice = self.data[..., start:stop]

        fluo_copy._data = time_slice.copy()

        return fluo_copy

    def time_mean(self, ignore_nan=False):
        """Mean of the fluorescence signal for each time period."""
        fluo_copy = self.copy(read_only=True)

        if ignore_nan:
            fluo_copy._data = np.nanmean(
                fluo_copy.data, axis=fluo_copy.data.ndim - 1
            )[..., np.newaxis]
        else:
            fluo_copy._data = np.mean(
                fluo_copy.data, axis=fluo_copy.data.ndim - 1
            )[..., np.newaxis]

        return fluo_copy

    def time_std(self, ignore_nan=False):
        """Standard deviation of the fluorescence signal for each time period."""
        fluo_copy = self.copy(read_only=True)

        if ignore_nan:
            fluo_copy._data = np.nanstd(
                fluo_copy.data, axis=fluo_copy.data.ndim - 1
            )[..., np.newaxis]
        else:
            fluo_copy._data = np.std(
                fluo_copy.data, axis=fluo_copy.data.ndim - 1
            )[..., np.newaxis]

        return fluo_copy

    def copy(self, read_only=False):
        """Get a deep copy.

        Parameters
        ----------
        read_only : bool, default False
            Whether to get a read-only copy of the underlying `data` array.
            Getting a read-only copy is much faster and should be used if a
            large number of copies need to be created.

        """
        if read_only:
            # Get a read-only view of the fluo array
            # This is much faster than creating a full copy
            read_only_fluo = self.data.view()
            read_only_fluo.flags.writeable = False

            deepcopy_memo = {id(self._data): read_only_fluo}
            copy_ = deepcopy(self, deepcopy_memo)
        else:
            copy_ = deepcopy(self)

        return copy_

    def _get_cells_from_mask(self, mask):
        cell_subset = self.copy(read_only=True)

        cell_subset._cell_vec = self.cell_vec[mask].copy()
        cell_subset._data = self.data[..., mask, :].copy()

        assert cell_subset.data.ndim == self.data.ndim
        assert cell_subset.num_cells == np.sum(mask)

        return cell_subset

    def positive_part(self):
        """Set the negative part of data to zero."""
        if self.is_positive_clipped:
            raise ValueError("Instance is already positive clipped.")
        fluo_copy = self.copy(read_only=False)

        with fluo_copy.unlock():
            fluo_copy.data[fluo_copy.data < 0] = 0
            fluo_copy._is_positive_clipped = True

        return fluo_copy


class RawFluorescence(Fluorescence):
    """Fluorescence timeseries from a full imaging session.

    Not divided into trials.

    """

    def __init__(self, fluorescence_array, timestep_width):
        fluorescence_array = np.asarray(fluorescence_array)
        assert fluorescence_array.ndim == 2

        super().__init__(fluorescence_array, timestep_width)

    def z_score(self):
        """Convert to Z-score."""
        if self.is_z_score:
            raise ValueError("Instance is already a Z-score")
        else:
            z_score = self.data - self.time_mean().data
            z_score /= z_score.time_std().data

            with self.unlock():
                self.data = z_score
                self._is_z_score = True

    def cut_by_trials(
        self,
        trial_timetable,
        num_baseline_frames=None,
        both_ends_baseline=False,
    ):
        """Divide fluorescence traces up into equal-length trials.

        Parameters
        ----------
        trial_timetable : pd.DataFrame-like
            A DataFrame-like object with 'Start' and 'End' items for the start
            and end frames of each trial, respectively.

        Returns
        -------
        trial_fluorescence : TrialFluorescence

        """
        if ("Start" not in trial_timetable) or ("End" not in trial_timetable):
            raise ValueError(
                "Could not find `Start` and `End` in trial_timetable."
            )

        if (num_baseline_frames is None) or (num_baseline_frames < 0):
            num_baseline_frames = 0

        # Slice the RawFluorescence up into trials.
        trials = []
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
            if both_ends_baseline:
                end = int(end) + num_baseline_frames
            else:
                end = int(end)

            trials.append(self.data[..., start:end])
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
            for i in range(len(trials)):
                trials[i] = trials[i][..., :min_num_frames]

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
                trial_num = np.arange(0, len(trials))

        # Construct TrialFluorescence and return it.
        trial_fluorescence = TrialFluorescence(
            np.asarray(trials), trial_num, self.timestep_width,
        )
        trial_fluorescence._is_z_score = self.is_z_score
        trial_fluorescence._is_dff = self.is_dff
        trial_fluorescence._baseline_duration = (
            num_baseline_frames * self.timestep_width
        )
        trial_fluorescence._both_ends_baseline = both_ends_baseline

        # Check that trial_fluorescence was constructed correctly.
        assert trial_fluorescence.num_cells == self.num_cells
        assert trial_fluorescence.num_timesteps == min_num_frames
        assert trial_fluorescence.num_trials == len(trials)

        return trial_fluorescence

    def plot(self, ax=None, **pltargs):
        if ax is not None:
            ax = plt.gca()

        ax.imshow(self.data, **pltargs)

        return ax

    def apply_quality_control(self, inplace=False):
        raise NotImplementedError


class TrialFluorescence(Fluorescence, TrialDataset):
    """Fluorescence timeseries divided into trials."""

    def __init__(self, fluorescence_array, trial_num, timestep_width):
        fluorescence_array = np.asarray(fluorescence_array)
        assert fluorescence_array.ndim == 3
        assert fluorescence_array.shape[0] == len(trial_num)

        super().__init__(fluorescence_array, timestep_width)

        self._baseline_duration = 0
        self._both_ends_baseline = False
        self._trial_num = np.asarray(trial_num)

    @property
    def time_vec(self):
        time_vec_without_baseline = super().time_vec
        return time_vec_without_baseline - self._baseline_duration

    def plot(
        self,
        ax=None,
        fill_mean_pm_std=True,
        highlight_non_baseline=False,
        **pltargs
    ):
        if ax is None:
            ax = plt.gca()

        if self.num_cells == 1:
            # If there is only one cell, make a line plot
            alpha = pltargs.pop("alpha", 1)

            fluo_mean = self.trial_mean().data[0, 0, :]
            fluo_std = self.trial_std().data[0, 0, :]

            if fill_mean_pm_std:
                ax.fill_between(
                    self.time_vec,
                    fluo_mean - fluo_std,
                    fluo_mean + fluo_std,
                    label="Mean $\pm$ SD",
                    alpha=alpha * 0.6,
                    **pltargs,
                )

            ax.plot(self.time_vec, fluo_mean, alpha=alpha, **pltargs)
            if highlight_non_baseline:
                stim_start = self.time_vec[0] + self._baseline_duration
                if self._both_ends_baseline:
                    stim_end = self.time_vec[-1] - self._baseline_duration
                else:
                    stim_end = self.time_vec[-1]
                ax.axvspan(
                    stim_start,
                    stim_end,
                    color="gray",
                    alpha=0.3,
                    label="Stimulus",
                )
            ax.set_xlabel("Time (s)")
            if self.is_z_score:
                ax.set_ylabel("DF/F (Z-score)")
            else:
                ax.set_ylabel("DF/F")
            ax.legend()
        else:
            # If there are many cells, just show the mean as a matrix.
            ax.imshow(self.trial_mean().data[0, ...], **pltargs)

        return ax

    def apply_quality_control(self, inplace=False):
        raise NotImplementedError

    def _get_trials_from_mask(self, mask):
        trial_subset = self.copy(read_only=True)

        trial_subset._trial_num = trial_subset._trial_num[mask].copy()
        trial_subset._data = trial_subset.data[mask, ...].copy()

        return trial_subset

    def trial_mean(self, ignore_nan=False):
        """Get the mean fluorescence for each cell across all trials.

        Parameters
        ----------
        ignore_nan : bool, default False
            Whether to return the `mean` or `nanmean` for each cell.

        Returns
        -------
        trial_mean : TrialFluoresence
            A new `TrialFluorescence` object with the mean across trials.

        See Also
        --------
        `trial_std()`

        """
        trial_mean = self.copy(read_only=True)

        trial_mean._trial_num = np.asarray([np.nan])

        if ignore_nan:
            trial_mean._data = np.nanmean(self.data, axis=0)[np.newaxis, :, :]
        else:
            trial_mean._data = self.data.mean(axis=0)[np.newaxis, :, :]

        return trial_mean

    def trial_std(self, ignore_nan=False):
        """Get the standard deviation of the fluorescence for each cell across trials.

        Parameters
        ----------
        ignore_nan : bool, default False
            Whether to return the `std` or `nanstd` for each cell.

        Returns
        -------
        trial_std : TrialFluorescence
            A new `TrialFluorescence` object with the standard deviation across
            trials.

        See Also
        --------
        `trial_mean()`

        """
        trial_std = self.copy(read_only=True)

        trial_std._trial_num = np.asarray([np.nan])

        if ignore_nan:
            trial_std._data = np.nanstd(self.data, axis=0)[np.newaxis, :, :]
        else:
            trial_std._data = self.data.std(axis=0)[np.newaxis, :, :]

        return trial_std
