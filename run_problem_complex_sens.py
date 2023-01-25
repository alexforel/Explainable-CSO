# -*- coding: utf-8 -*-

"""
Sensitivity analysis to the number of uncertain parameters
and the number of decisions in the decision problem.
"""

# Import packages
import sys
import gurobipy as GBP
import pandas as pd
import os
# Import custom functions
from src.Simulator import Simulator
from src.setup_experiment import run_sensitivity_experiment

dirname = os.path.dirname(__file__)
# Create gurobi environment
gurobiEnv = GBP.Env()
gurobiEnv.setParam('TimeLimit', 3600)
gurobiEnv.setParam('Threads', 8)
gurobiEnv.start()

# ---- Create simulation setting with synthetic data ----
experimentName = sys.argv[1]
experimentList = ['newsvendor', 'path']
try:
    assert experimentName in experimentList
except AssertionError:
    print('Incorrect experiment name: ', experimentName)
    print('Experiment name should be in ', experimentList)
    raise

nbSamples = 100
nbTrees = 100
nbNeighbors = 10
NB_REPETITIONS = 100

if experimentName == 'newsvendor':
    # Number of products
    sensList = [5, 20, 50, 100]
elif experimentName == 'path':
    # Grid length
    sensList = [3, 5, 10, 20]

# ---- Run simulations ----
resultDataframe = pd.DataFrame()
print('---- Running experiment: {} ----'.format(experimentName))
for problemSens in sensList:
    print('  -- Problem complexity parameter: {}'.format(problemSens))
    for i in range(NB_REPETITIONS):
        simulator = Simulator(experimentName, gurobiEnv,
                              nbSamples=nbSamples, verbose=False,
                              problemSens=problemSens)
        x_init = simulator.sample()
        x_alt = simulator.sample()
        # ----------------- RANDOM FORESTS ---------------------
        z_opt, z_alt, x_r, x_a, t_r, t_a = run_sensitivity_experiment(
            simulator, x_init, x_alt, simulator.get_model, gurobiEnv,
            prescriptorType='rf', nbTrees=nbTrees)
        partialResultDf = pd.DataFrame(
            {'experiment': [experimentName],
             'problemSens': [problemSens],
             'nbIndex': [nbSamples],
             'simIndex': [i],
             'method': ['rf'],
             'nbTrees': [nbTrees],
             'nbNeighbors': [0],
             'x_init': [x_init],
             'x_alt': [x_alt],
             'z_opt': [z_opt],
             'z_alt': [z_alt],
             'x_rel': [x_r],
             'x_abs': [x_a],
             'solvetime_relative': [t_r],
             'solvetime_absolute': [t_a]})
        resultDataframe = pd.concat(
            [resultDataframe, partialResultDf],
            ignore_index=True, axis=0)

        # -----------------       KNN      ---------------------
        z_opt, z_alt, x_r, x_a, t_r, t_a = run_sensitivity_experiment(
            simulator, x_init, x_alt, simulator.get_model, gurobiEnv,
            prescriptorType='knn', nbNeighbors=nbNeighbors)
        partialResultDf = pd.DataFrame(
            {'experiment': [experimentName],
             'problemSens': [problemSens],
             'nbIndex': [nbSamples],
             'simIndex': [i],
             'method': ['knn'],
             'nbTrees': [0],
             'nbNeighbors': [nbNeighbors],
             'x_init': [x_init],
             'x_alt': [x_alt],
             'z_opt': [z_opt],
             'z_alt': [z_alt],
             'x_rel': [x_r],
             'x_abs': [x_a],
             'solvetime_relative': [t_r],
             'solvetime_absolute': [t_a]})
        resultDataframe = pd.concat(
            [resultDataframe, partialResultDf],
            ignore_index=True, axis=0)

# Save results fo csv file
folderName = os.path.join(dirname, "output",)
os.makedirs(folderName, exist_ok=True)
resultDataframe.to_csv(folderName+'/'+experimentName+'_prob_complex_sens.csv')
