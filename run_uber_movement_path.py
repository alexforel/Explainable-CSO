# -*- coding: utf-8 -*-

"""
Explaining contextual shortest path with Uber movement data:
        - Get context and uncertain parameters.
        - Train a random forest prescriptor and derive optimal decision.
        - Explain decision through counterfactual explanations.
"""
# Import packages
import seaborn as sns
import os as os
import numpy as np
import gurobipy as GBP
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
# Import custom functions
from src.Simulator import Simulator
from src.RfPrescriptor import RfPrescriptor
from src.UberPath import get_node_list, get_census_dict, get_gps_coordinates

# Create gurobi environment
gurobiEnv = GBP.Env()
gurobiEnv.start()

# ---- Create simulation setting with synthetic data ----
experimentName = 'uber'
# Create a Simulator object to store the data and counterfactuals
simulator = Simulator(experimentName, gurobiEnv)
x_init = simulator.sample()
simulator.X_train = simulator.X_train[0:1000]
simulator.Y_train = simulator.Y_train[0:1000]

# ----- Setup experiment -----
# Read the GPS coordinates of the nodes
# and create the road network as a directed graph
dirname = os.path.dirname(__file__)
dirPath = os.path.join(dirname, "data", "census",)
nodeList = get_node_list()
census = get_census_dict()
centroidGps = get_gps_coordinates(dirPath, census)

# Create list of edges of LA graph
n_rows, n_cols = simulator.A_mat.shape
edgeList = []
for c in range(n_cols):
    for r in range(n_rows):
        if simulator.A_mat[r, c] == -1:
            start = census[nodeList[r]]
        elif simulator.A_mat[r, c] == 1:
            end = census[nodeList[r]]
    edgeList.append((start, end))

# Create graph
G = nx.DiGraph()
G.add_edges_from(edgeList)

# Train random forest prescriptor
rfPrescriptor = RfPrescriptor(simulator.X_train, simulator.Y_train,
                              simulator.get_model, gurobiEnv,
                              nbTrees=100, max_depth=4, isScaled=True)

# ------ Solve decision problem ------
# Set start and end nodes
startNode, endNode = ['206032', '209401']
simulator.set_start_end_nodes(startNode, endNode, census, nodeList)
# Solve CSO problem
z_opt = rfPrescriptor.solve_cso_problem(x_init, simulator)
# Read path
opt_path = []
for i in z_opt:
    if z_opt[i] == 1:
        opt_path.append(edgeList[i])
altPath = [('206032', '206031'),
           ('206031', '226002'),
           ('226002', '207900'),
           ('207900', '207710'),
           ('207710', '210010'),
           ('210010', '209520'),
           ('209520', '209401')]
# Show graph and optimal paths
fig, ax = plt.subplots(figsize=[10, 10])
nx.draw_networkx_nodes(G, pos=centroidGps, node_size=30, node_color='black',
                       alpha=.9)
nx.draw_networkx_edges(G, pos=centroidGps, edge_color='gray', alpha=.5)
nx.draw_networkx_labels(G, centroidGps, verticalalignment='top')
# Add start and end nodes in blue
nx.draw_networkx_nodes([startNode, endNode], pos=centroidGps,
                       node_size=50, node_color='blue')
# Add paths
nx.draw_networkx_edges(G, pos=centroidGps, edgelist=opt_path,
                       edge_color='green', alpha=1, width=5)
nx.draw_networkx_edges(G, pos=centroidGps, edgelist=altPath,
                       edge_color='orange', alpha=1, width=3)

# Plot on a map
BBox = (-118.285, -118.22, 34.028, 34.065)
map_img = plt.imread(os.path.join(dirname, "data", 'map.png'))
# Show graph and optimal paths
fig, ax = plt.subplots(figsize=[9, 15])
nx.draw_networkx_nodes(G, pos=centroidGps, node_size=30, node_color='black',
                       alpha=.9)
nx.draw_networkx_edges(G, pos=centroidGps, edge_color='gray',
                       alpha=.7, width=2.0)
# Add start and end nodes in blue
nx.draw_networkx_nodes([startNode, endNode], pos=centroidGps,
                       node_size=80, node_color='blue')
# Add paths
nx.draw_networkx_edges(G, pos=centroidGps, edgelist=opt_path,
                       edge_color='blue', alpha=1, width=4)
nx.draw_networkx_edges(G, pos=centroidGps, edgelist=altPath,
                       edge_color='green', alpha=1, width=4)
# Setting limits for the plot
ax.set_xlim(BBox[0], BBox[1])
ax.set_ylim(BBox[2], BBox[3])
# Showing the image behind the points
ax.imshow(map_img, zorder=0, extent=BBox, aspect='equal', alpha=0.9)
outputFilePath = os.path.join(dirname, "output", "figs", "uberPath.pdf")
plt.savefig(outputFilePath, bbox_inches='tight')
plt.show()

# ------ Explain decisions ------
# Convert alt path to alt decision
z_alt = dict()
for i, edge in enumerate(edgeList):
    if edge in altPath:
        z_alt[i] = 1
    else:
        z_alt[i] = 0
# Solve explanation problems
x_rel, time_soft = rfPrescriptor.solve_explanation_problem(
        x_init, z_opt, z_alt,
        isRandomForestPrescriptor=True,
        getAbsoluteExplanation=False,
        useIsolationForest=False,
        verbose=True,
        featuresType=simulator.reader.featuresType,
        featuresPossibleValues=simulator.reader.featuresPossibleValues,
        oneHotEncoding=simulator.reader.oneHotEncoding)
x_abs, time_hard = rfPrescriptor.solve_explanation_problem(
        x_init, z_opt, z_alt,
        isRandomForestPrescriptor=True,
        getAbsoluteExplanation=True,
        useIsolationForest=False,
        verbose=True,
        featuresType=simulator.reader.featuresType,
        featuresPossibleValues=simulator.reader.featuresPossibleValues,
        oneHotEncoding=simulator.reader.oneHotEncoding)

# Print results
print('Initial context: ', x_init)
print('Relative explanation: ', x_rel)
print('Absolute explanation: ', x_abs)
# Plot init and explanations as heatmap
x_array = np.concatenate((np.array(x_init), x_rel, x_abs), axis=0)
newcmp = LinearSegmentedColormap.from_list(
    "", ["white", "royalblue"])
fig, ax = plt.subplots(figsize=[8, 4])
sns.heatmap(x_array, linewidths=0, cmap=newcmp, vmin=0, vmax=1)
ax.set_xticklabels(x_init.columns.values.tolist(), rotation=45, ha='right')
