# -*- coding: utf-8 -*-

"""
Generate explanations with spurious features:
used to analyze the correlation and the percentage
of relevant feature changes.
"""

# Import packages
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
experimentName = 'newsvendor'
# Experiment parameters
nbSamplesList = [50, 100, 200, 400, 600, 800]
nbTreesList = [100]
depthList = [4]
NB_REPETITIONS = 100
NB_SPURIOUS = 2

# ---- Run simulations ----
print('---- Running experiment: {} ----'.format(experimentName))
for nbSamples in nbSamplesList:
    resultDataframe = pd.DataFrame()
    print('\n  -- Nb. samples: {}'.format(nbSamples))
    for i in range(NB_REPETITIONS):
        print("     Simulation: {} out of {}.".format(
            i+1, NB_REPETITIONS), end='\r')
        simulator = Simulator(experimentName, gurobiEnv,
                              nbSamples=nbSamples, verbose=False,
                              nbSpurious=NB_SPURIOUS)
        x_init = simulator.sample()
        x_alt = simulator.sample()
        # ----------------- RANDOM FORESTS ---------------------
        for nbTrees in nbTreesList:
            for treeDepth in depthList:
                z_opt, z_alt, x_r, x_a, t_r, t_a = run_sensitivity_experiment(
                    simulator, x_init, x_alt, simulator.get_model,
                    gurobiEnv, prescriptorType='rf', nbTrees=nbTrees,
                    max_depth=treeDepth)
                partialResultDf = pd.DataFrame(
                    {'experiment': [experimentName],
                     'nbSamples': [nbSamples],
                     'simIndex': [i],
                     'method': ['rf'],
                     'nbTrees': [nbTrees],
                     'treeDepth': [treeDepth],
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
    resultDataframe.to_csv(folderName
                           + '/spurious_{}_results_n_{}.csv'.format(
                                experimentName, nbSamples))
