"""Objects for interacting with running speed datasets."""

__all__ = ("RunningSpeed")

import numpy as np
import matplotlib.pyplot as plt

from .base_objects import TimeseriesDataset
from .util import (
    robust_range, ROBUST_PLOT_RANGE_DEFAULT_HALF_WIDTH
)

class RunningSpeed(TimeseriesDataset):
    def __init__(self, running_speed: np.ndarray, timestep_width: float):
        running_speed = np.asarray(running_speed)
        assert running_speed.ndim == 1

        super().__init__(timestep_width)
        self.data = running_speed

    @property
    def num_timesteps(self):
        """Number of timesteps in RunningSpeed dataset."""
        return len(self.data)

    def get_frame_range(self, start: int, stop: int = None):
        window = self.copy()
        if stop is not None:
            window.data = window.data[start:stop, :].copy()
        else:
            window.data = window.data[start, :].copy()

        return window

    def time_mean(self, ignore_nan=False):
        """Mean of the eye tracking data for each time period."""
        raise NotImplementedError('Implementation may be incorrect')
        running_copy = self.copy()

        if ignore_nan:
            running_copy.data = np.nanmean(
                running_copy.data, axis=running_copy.data.ndim - 1
            )[..., np.newaxis]
        else:
            running_copy.data = np.mean(
                running_copy.data, axis=running_copy.data.ndim - 1
            )[..., np.newaxis]

        return running_copy

    def time_std(self, ignore_nan=False):
        """Standard deviation of the eye tracking data for each time period."""
        raise NotImplementedError('Implementation may be incorrect')
        running_copy = self.copy()

        if ignore_nan:
            running_copy.data = np.nanstd(
                running_copy.data, axis=running_copy.data.ndim - 1
            )[..., np.newaxis]
        else:
            running_copy.data = np.std(
                running_copy.data, axis=running_copy.data.ndim - 1
            )[..., np.newaxis]

        return running_copy

    def plot(self, robust_range_=False, ax=None, **pltargs):
        if ax is None:
            ax = plt.gca()

        if robust_range_:
            ax.axhspan(
                *robust_range(
                    self.data, half_width=1.5, center="median", spread="iqr"
                ),
                color="gray",
                label="Median $\pm$ 1.5 IQR",
                alpha=0.5,
            )
            ax.legend()

        ax.plot(self.time_vec, self.data, **pltargs)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Running speed")

        if robust_range_:
            ax.set_ylim(
                robust_range(
                    self.data, half_width=ROBUST_PLOT_RANGE_DEFAULT_HALF_WIDTH
                )
            )

        return ax

    def apply_quality_control(self, inplace=False):
        super().apply_quality_control(inplace)

