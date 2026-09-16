"""CSV layout and feature configuration shared by the ST-GeoNet workflow."""
import csv
from pathlib import Path
import pandas as pd

PREFIX_COLUMNS = ('Gauss_X', 'Gauss_Y', 'longitude', 'latitude', 'speed', 'direction')
RAW_TAIL_COLUMNS = ('time', 'tag')
FEATURE_TAIL_COLUMNS = ('time', 'degree', 'tag')
RESERVED_COLUMNS = set(PREFIX_COLUMNS + FEATURE_TAIL_COLUMNS + ('point_id', 'index', 'Unnamed: 0'))

def normalize_extra_features(names=None):
    """Validate an ordered list of additional numeric feature column names."""
    if isinstance(names, str):
        raise ValueError('extra_features must be a list of column names, not a single string.')
    result = list(names or [])
    if any(not isinstance(name, str) or not name or name != name.strip() for name in result):
        raise ValueError('Extra feature names must be nonempty strings without surrounding spaces.')
    if len(set(result)) != len(result):
        raise ValueError('Extra feature names must be unique.')
    reserved = [name for name in result if name in RESERVED_COLUMNS]
    if reserved:
        raise ValueError(f'Fixed CSV columns cannot be selected as extra features: {reserved}.')
    return result

def get_extra_columns(columns, prepared=False):
    """Validate fixed prefix/suffix positions and return the middle feature names.

    The CSV index is excluded from columns. Raw/sampled rows end with time, tag;
    prepared rows end with time, degree, tag. Additional columns follow direction.
    """
    columns = list(columns)
    tail = FEATURE_TAIL_COLUMNS if prepared else RAW_TAIL_COLUMNS
    if len(set(columns)) != len(columns):
        raise ValueError('CSV column names must be unique.')
    if (len(columns) < len(PREFIX_COLUMNS) + len(tail)
            or tuple(columns[:len(PREFIX_COLUMNS)]) != PREFIX_COLUMNS
            or tuple(columns[-len(tail):]) != tail):
        layout = ','.join(PREFIX_COLUMNS + ('[extra numeric columns]',) + tail)
        raise ValueError(f'Expected CSV columns after the index column: {layout}.')
    return normalize_extra_features(columns[len(PREFIX_COLUMNS):-len(tail)])

def read_trajectory_csv(path, prepared=False):
    """Read a UTF-8 indexed trajectory CSV without losing additional columns."""
    path = Path(path)
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), None)
    if not header or len(header) < 2:
        raise ValueError(f'{path}: expected an index column and named trajectory columns.')
    if header[0] and header[0] in header[1:]:
        raise ValueError(f'{path}: the index-column name must not duplicate a data-column name.')
    try:
        get_extra_columns(header[1:], prepared=prepared)
        frame = pd.read_csv(path, index_col=0)
        get_extra_columns(frame.columns, prepared=prepared)
    except ValueError as error:
        raise ValueError(f'{path}: {error}') from error
    return frame

def make_feature_config(extra_features=None, motion_source='derived'):
    """Describe the ordered network input and its normalization conventions."""
    extras = normalize_extra_features(extra_features)
    if motion_source not in ('derived', 'csv'):
        raise ValueError("motion_source must be 'derived' or 'csv'.")
    motion = ['derived_distance', 'derived_direction'] if motion_source == 'derived' else ['speed', 'direction']
    return {
        'version': 1,
        'extra_features': extras,
        'motion_source': motion_source,
        'input_dim': 6 + len(extras),
        'input_order': ['Gauss_X', 'Gauss_Y', 'zero'] + motion + extras + ['degree'],
        'motion_and_extra_normalization': 'per_trajectory_minmax_eps_1e-6',
        'degree_source': 'prepared_csv',
    }
