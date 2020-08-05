import numpy as np


class SliceParseError(Exception):
    pass


def try_parse_positionals_as_slice_like(*args):
    """Try to parse positional arguments as a slice-like int or pair of ints.

    Output can be treated as a `(start, stop)` range (where `stop` is optional)
    on success, and can be treated as a boolean mask if a `SliceParseError` is
    raised.

    Returns
    -------
    slice_like : [int] or [int, int]

    Raises
    ------
    SliceParseError
        If positional arguments are a boolean mask, not a slice.
    TypeError
        If positional arguments are not bool-like or int-like.
    ValueError
        If positional arguments are empty or have more than two entries.

    """
    flattened_args = np.asarray(args).flatten()
    if len(flattened_args) == 0:
        raise ValueError("Empty positional arguments")
    elif _is_bool(flattened_args[0]):
        raise SliceParseError("Cannot parse bool positionals as slice.")
    elif int(flattened_args[0]) != flattened_args[0]:
        raise TypeError(
            "Expected positionals to be bool-like or int-like, "
            "got type {} instead".format(flattened_args.dtype)
        )
    elif (len(flattened_args) > 0) and (len(flattened_args) <= 2):
        # Positional arguments are a valid slice-like int or pair of ints
        return flattened_args.tolist()
    else:
        # Case: positionals are not bool and are of the wrong length
        raise ValueError(
            "Positionals of length {} cannot be parsed as slice-like".format(
                len(flattened_args)
            )
        )


def _is_bool(x):
    return isinstance(x, (bool, np.bool, np.bool8, np.bool_))


def validate_vector_mask_length(mask, expected_length):
    if np.ndim(mask) != 1:
        raise ValueError(
            "Expected mask to be vector-like, got "
            "{}D array instead".format(np.ndim(mask))
        )

    mask = np.asarray(mask).flatten()
    if len(mask) != expected_length:
        raise ValueError(
            "Expected mask of length {}, got mask of "
            "length {} instead.".format(len(mask), expected_length)
        )

    return mask


def get_vector_mask_from_range(values_to_mask, start, stop=None):
    """Unmask all values within a range."""
    if stop is not None:
        mask = values_to_mask >= start
        mask &= values_to_mask < stop
    else:
        mask = values_to_mask == start
    return mask


def robust_range(
    values, half_width=2, center="median", spread="interquartile_range"
):
    """Get a range around a center point robust to outliers."""
    if center == "median":
        center_val = np.nanmedian(values)
    elif center == "mean":
        center_val = np.nanmean(values)
    else:
        raise ValueError(
            "Unrecognized `center` {}, expected "
            "`median` or `mean`.".format(center)
        )

    if spread in ("interquartile_range", "iqr"):
        lower_quantile, upper_quantile = np.percentile(
            _stripnan(values), (25, 75)
        )
        spread_val = upper_quantile - lower_quantile
    elif spread in ("standard_deviation", "std"):
        spread_val = np.nanstd(values)
    else:
        raise ValueError(
            "Unrecognized `spread` {}, expected "
            "`interquartile_range` (`iqr`) or `standard_deviation` (`std`)".format(
                spread
            )
        )

    lower_bound = center_val - half_width * spread_val
    upper_bound = center_val + half_width * spread_val

    return (lower_bound, upper_bound)


def _stripnan(values):
    values_arr = np.asarray(values).flatten()
    return values_arr[~np.isnan(values_arr)]


ROBUST_PLOT_RANGE_DEFAULT_HALF_WIDTH = 3
