"""Sample cleaned, projected trajectory CSV files while preserving row order.

Input CSV files contain an index column, the fixed coordinate and motion
columns, optional numeric attributes after direction, and the final time and
tag columns. All attributes are carried into each sampled trajectory view.
Input and output directories are resolved relative to this script.
"""
import os
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from trajectory_schema import read_trajectory_csv

# Maximum number of points in each sampled trajectory view.
N = 4096
mode = 'test'
input_folder = SCRIPT_DIR / 'Unsampled_data' / mode
output_folder = SCRIPT_DIR / 'Sampled_data' / mode
os.makedirs(output_folder, exist_ok=True)

# Read validated columns and retain their names and order in sampled outputs.
for filename in os.listdir(input_folder):
    if re.match(r'.*\.csv$', filename):
        path = os.path.join(input_folder, filename)
        data_frame = read_trajectory_csv(path, prepared=False)
        Header = data_frame.columns.tolist()
        data = data_frame.values
        # Remove stationary points using the supplied speed column.
        data = np.delete(data, data[:, 4] == 0, 0)
        number_data = len(data)

        # Short trajectories are saved without padding; the loader pads them later.
        if number_data <= N:
            output_path = os.path.join(output_folder, filename)
            pd.DataFrame(data).to_csv(output_path, index=True, header=Header)
            continue

        # Mark an initial set of approximately evenly spaced temporal indices.
        data = np.hstack((data, np.zeros((number_data, 1))))
        S = number_data // np.ceil(number_data / N).astype(int)
        data[np.arange(0, S * np.ceil(number_data / N).astype(int), np.ceil(number_data / N).astype(int)), -1] = 1

        # Track unselected rows and the remaining capacity of the current view.
        F = data[:, -1] == 0
        I = np.column_stack((np.where(F)[0], np.zeros(np.sum(F), dtype=int)))
        r_d = [number_data - S]
        r_N = [N - S]
        S = [S]

        # Repeatedly fill the remaining capacity from unselected temporal indices.
        while N != S[-1]:
            sample_rate = np.ceil(r_d[-1] / r_N[-1]).astype(int)
            S = np.append(S, r_d[-1] // sample_rate)
            r_d = np.append(r_d, r_d[-1] - S[-1])
            r_N = np.append(r_N, r_N[-1] - S[-1])
            I[:sample_rate * S[-1]:sample_rate, -1] = 1
            data[I[I[:, -1] == 1, 0], -1] = 1
            F = I[:, -1] == 0
            I = np.column_stack((I[F, 0], np.zeros(np.sum(F), dtype=int)))
            if r_N[-1] == 0:
                break

        # Shift the selection mask by one row to create overlapping sampled views.
        # Boolean selection keeps each saved view in the original temporal order.
        for j in range(int(np.ceil(len(data) / N)) + 1):
            output_path = os.path.join(output_folder, f'{filename}_split_{j}.csv')
            pd.DataFrame(data[data[:, -1] == 1, :-1]).to_csv(output_path, index=True, header=Header)
            data[:, -1] = np.roll(data[:, -1], 1)
