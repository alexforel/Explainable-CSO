# -*- coding: utf-8 -*-

"""
Compare the dual reformulation and the iterative algorithm
to generate absolute explanations when the decision
problem is linear.
"""

# Import packages
import gurobipy as GBP
import pandas as pd
import os
# Import custom functions
from src.Simulator import Simulator
from src.setup_experiment import run_sensitivity_experiment
from src.setup_experiment import assert_all_elements_equal

dirname = os.path.dirname(__file__)
# Create gurobi environment
gurobiEnv = GBP.Env()
gurobiEnv.setParam('TimeLimit', 3600)
gurobiEnv.setParam('Threads', 8)
gurobiEnv.start()

# ---- Create simulation setting with synthetic data ----
experimentName = 'path'
nbSamplesList = [50, 100, 200]
useDualList = [False, True]

nbTrees = 100
nbNeighbors = 10
NB_REPETITIONS = 100
# ---- Run simulations ----
random_state = 0
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
        random_state = random_state + 1
        for useDualForm in useDualList:
            z_opt, z_alt, x_r, x_a, rTime, aTime = run_sensitivity_experiment(
                simulator, x_init, x_alt, simulator.get_model,
                gurobiEnv, prescriptorType='rf', nbTrees=nbTrees,
                useDual=useDualForm, random_state=random_state)
            partialResultDf = pd.DataFrame(
                {'experiment': [experimentName],
                 'nbSamples': [nbSamples],
                 'simIndex': [i],
                 'method': ['rf'],
                 'nbTrees': [nbTrees],
                 'nbNeighbors': [0],
                 'useDual': [useDualForm],
                 'x_init': [x_init],
                 'x_alt': [x_alt],
                 'z_opt': [z_opt],
                 'z_alt': [z_alt],
                 'x_rel': [x_r],
                 'x_abs': [x_a],
                 'solvetime_relative': [rTime],
                 'solvetime_absolute': [aTime]})
            resultDataframe = pd.concat(
                [resultDataframe, partialResultDf],
                ignore_index=True, axis=0)
        # - Check that formulations have same results
        fieldsToCheck = ['x_init', 'x_alt', 'z_opt', 'z_alt']
        for field in fieldsToCheck:
            try:
                assert_all_elements_equal(resultDataframe, field, 2)
            except AssertionError:
                print('Error: same simulations have different results!')
                print('field: ', field)
                print('Last 2 values: ',
                      resultDataframe[field].tolist()[-2:])
                raise
        # -----------------       KNN      ---------------------
        for useDualForm in useDualList:
            z_opt, z_alt, x_r, x_a, rTime, aTime = run_sensitivity_experiment(
                simulator, x_init, x_alt, simulator.get_model,
                gurobiEnv, prescriptorType='knn', nbNeighbors=nbNeighbors,
                useDual=useDualForm)
            partialResultDf = pd.DataFrame(
                {'experiment': [experimentName],
                 'nbSamples': [nbSamples],
                 'simIndex': [i],
                 'method': ['knn'],
                 'nbTrees': [0],
                 'nbNeighbors': [nbNeighbors],
                 'useDual': [useDualForm],
                 'x_init': [x_init],
                 'x_alt': [x_alt],
                 'z_opt': [z_opt],
                 'z_alt': [z_alt],
                 'x_rel': [x_r],
                 'x_abs': [x_a],
                 'solvetime_relative': [rTime],
                 'solvetime_absolute': [aTime]})
            resultDataframe = pd.concat(
                [resultDataframe, partialResultDf],
                ignore_index=True, axis=0)
        # - Check that formulations have same results
        fieldsToCheck = ['x_init', 'x_alt', 'z_opt', 'z_alt']
        for field in fieldsToCheck:
            try:
                assert_all_elements_equal(resultDataframe, field, 2)
            except AssertionError:
                print('Error: same simulations have different results!')
                print('field: ', field)
                print('Last 2 values: ',
                      resultDataframe[field].tolist()[-2:])
                raise
    # Save results fo csv file
    folderName = os.path.join(dirname, "output",)
    os.makedirs(folderName, exist_ok=True)
    resultDataframe.to_csv(folderName
                           + '/{}_dual_formulation_results_n_{}.csv'.format(
                                experimentName, nbSamples))
