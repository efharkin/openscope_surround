"""Objects for interacting with individual datasets.

For an introduction, see `base_objects.py`.

"""

__all__ = (
    "Dataset",
    "TimeseriesDataset",
    "TrialDataset",
    "Fluorescence",
    "EyeTracking",
    "RawFluorescence",
    "TrialFluorescence",
    "RawEyeTracking",
    "TrialEyeTracking",
    "RunningSpeed",
)

# Base classes for defining shared interfaces
from .base_objects import Dataset, TimeseriesDataset, TrialDataset
from .fluorescence import Fluorescence
from .eyetracking import EyeTracking

# Concrete classes
from .fluorescence import RawFluorescence, TrialFluorescence
from .eyetracking import RawEyeTracking, TrialEyeTracking
from .runningspeed import RunningSpeed
