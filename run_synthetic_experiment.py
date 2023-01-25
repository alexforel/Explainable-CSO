# -*- coding: utf-8 -*-

"""
Run repeated experiments with synthetic data.
"""

# Import packages
import sys
import gurobipy as GBP
import pandas as pd
import os
# Import custom functions
from src.Simulator import Simulator
from src.setup_experiment import run_sensitivity_experiment
from src.setup_experiment import shipment_filter_first_stage

dirname = os.path.dirname(__file__)
# Create gurobi environment
gurobiEnv = GBP.Env()
gurobiEnv.setParam('TimeLimit', 3600)
gurobiEnv.setParam('Threads', 8)
gurobiEnv.start()

# ---- Create simulation setting with synthetic data ----
experimentName = sys.argv[1]
experimentList = ['newsvendor', 'shipment', 'path', 'cvar-path']
try:
    assert experimentName in experimentList
except AssertionError:
    print('Incorrect experiment name: ', experimentName)
    print('Experiment name should be in ', experimentList)
    raise

nbSamplesList = [50, 100, 200]
nbTreesList = [100, 200]
nbNeighborsList = [10, 30]
NB_REPETITIONS = 100

# ---- Run simulations ----
print('---- Running experiment: {} ----'.format(experimentName))
for nbSamples in nbSamplesList:
    resultDataframe = pd.DataFrame()
    print('\n  -- Nb. samples: {}'.format(nbSamples))
    for i in range(NB_REPETITIONS):
        print("     Simulation: {} out of {}.".format(
            i+1, NB_REPETITIONS), end='\r')
        simulator = Simulator(experimentName, gurobiEnv,
                              nbSamples=nbSamples, verbose=False)
        x_init = simulator.sample()
        x_alt = simulator.sample()
        # ----------------- RANDOM FORESTS ---------------------
        for nbTrees in nbTreesList:
            z_opt, z_alt, x_r, x_a, t_r, t_a = run_sensitivity_experiment(
                simulator, x_init, x_alt, simulator.get_model, gurobiEnv,
                prescriptorType='rf', nbTrees=nbTrees)
            z_opt = shipment_filter_first_stage(experimentName, z_opt)
            z_alt = shipment_filter_first_stage(experimentName, z_alt)
            partialResultDf = pd.DataFrame(
                {'experiment': [experimentName],
                 'nbSamples': [nbSamples],
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
        for k in nbNeighborsList:
            z_opt, z_alt, x_r, x_a, t_r, t_a = run_sensitivity_experiment(
                simulator, x_init, x_alt, simulator.get_model, gurobiEnv,
                prescriptorType='knn', nbNeighbors=k)
            z_opt = shipment_filter_first_stage(experimentName, z_opt)
            z_alt = shipment_filter_first_stage(experimentName, z_alt)
            partialResultDf = pd.DataFrame(
                {'experiment': [experimentName],
                 'nbSamples': [nbSamples],
                 'simIndex': [i],
                 'method': ['knn'],
                 'nbTrees': [0],
                 'nbNeighbors': [k],
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
    resultDataframe.to_csv(folderName+'/'+experimentName
                           + '_synth_results_n_'+str(nbSamples)+'.csv')
