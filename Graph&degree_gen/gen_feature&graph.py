"""Generate cooperative-graph features from sampled, projected trajectories.

Input CSV files contain an index, the fixed coordinate and motion columns,
optional numeric attributes after direction, and the final time and tag
columns. Graphs are built in projected units before coordinates are scaled.
Feature outputs retain all attributes and insert degree immediately before
tag. Input and output directories are resolved relative to the project.
"""
import os
import re
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from triNEAUfunc import Delaunaycal

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from trajectory_schema import read_trajectory_csv

# Experiment settings; width uses the units of Gauss_X and Gauss_Y.
# N records the sampling size and is not used in this script.
N = 4096
width = 6
dataset = 'test'
mode = 'width_6_test'
plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["font.sans-serif"] = ['SimHei']
# Select the sampled subset and destination split independently.
input_folder = PROJECT_DIR / 'Sampling' / 'Sampled_data' / mode
output_folder_image = SCRIPT_DIR / 'Degree_data' / dataset / 'Image'
output_folder_feature = SCRIPT_DIR / 'Degree_data' / dataset / 'Feature'
output_folder_graph = SCRIPT_DIR / 'Degree_data' / dataset / 'Graph'
os.makedirs(output_folder_feature, exist_ok=True)
os.makedirs(output_folder_graph, exist_ok=True)
os.makedirs(output_folder_image, exist_ok=True)

for filename in os.listdir(input_folder):
    # Skip unrelated files before loading or generating trajectory outputs.
    if not re.match(r'.*\.csv$', filename):
        continue
    path = os.path.join(input_folder, filename)
    data = read_trajectory_csv(path, prepared=False)
    input_columns = data.columns.tolist()
    # Convert supported date-time strings to local-time Unix timestamps.
    tf_time = []
    for t in data["time"]:
        try:
            timeArray = time.strptime(t, "%Y-%m-%d %H:%M:%S")
        except:
            try:
                timeArray = time.strptime(t, "%Y/%m/%d %H:%M:%S")
            except:
                # Interpret time-only strings using the existing fallback date.
                t = "2022/5/1 " + t
                timeArray = time.strptime(t, "%Y/%m/%d %H:%M:%S")
        timestamp = time.mktime(timeArray)
        tf_time.append(timestamp)
    data['time'] = tf_time
    data = np.array(data, dtype=float)
    # Add random coordinate perturbations before triangulation.
    data[:, 0:2] += np.random.uniform(-0.01, 0.01, (len(data), 2))
    # Construct auxiliary points and redirect pruned triangulation edges.
    data_tri = Delaunaycal(data, width=width)
    acceptor = data_tri.adinter
    data_tri.calculatearea()
    tri_area, _, graph_edge = data_tri.area
    # Count retained edge incidences, cap at ten, and place degree before the label.
    degree = np.zeros((len(acceptor[2]), 1))
    for i in range(len(tri_area[0])):
        degree[range(len(acceptor[2])) == tri_area[0][i]] = tri_area[1][i]
    degree[degree >= 10] = 10
    data_with_degree = np.c_[acceptor[2], degree]
    data_with_degree = np.insert(data_with_degree, -2, data_with_degree[:, -1], axis=1)
    data_with_degree = np.delete(data_with_degree, -1, 1)

    # Translate both coordinate axes and divide them by one common spatial span.
    # Scale degree by its maximum value within the trajectory.
    data_with_degree[:, 0:2] = (data_with_degree[:, 0:2] - np.min(data_with_degree[:, 0:2], axis=0))
    data_with_degree[:, 0:2] /= np.max(data_with_degree[:, 0:2])
    data_with_degree[:, -2] /= np.max(data_with_degree[:, -2])

    # Remove the extra .csv suffix introduced by the sampling script.
    f = re.sub(r'\.csv_', '_', filename, count=1)
    f = re.search(r'(.*).csv', f)
    Header = input_columns[:-1] + ['degree'] + input_columns[-1:]
    feature_path = os.path.join(output_folder_feature, f.group(1) + '.csv')
    graph_path = os.path.join(output_folder_graph, f.group(1) + '.csv')
    pd.DataFrame(data_with_degree).to_csv(feature_path, index=True, header=Header)

    # Save the matching graph as two node-ID columns without a CSV header.
    np.savetxt(graph_path, graph_edge, delimiter=',')
