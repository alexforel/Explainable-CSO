# -*- coding: utf-8 -*-

"""
Sensitivity analysis of the number of features
of the contextual information.
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
experimentName = 'path'
nbTrees = 100
NB_REPETITIONS = 100
nbSamplesList = [100, 500, 1000, 5000]
nbFeaturesList = [5, 10, 25, 50, 100, 500]

# ---- Run simulations ----
for nbSamples in nbSamplesList:
    print('\n ---- n: {} ----'.format(nbSamples))
    for d_x in nbFeaturesList:
        print('\n   -- Nb features: {} '.format(d_x))
        resultDataframe = pd.DataFrame()
        for i in range(NB_REPETITIONS):
            print("     Simulation: {} out of {}.".format(
                i+1, NB_REPETITIONS), end='\r')
            simulator = Simulator(experimentName, gurobiEnv,
                                  nbSamples=nbSamples, verbose=False,
                                  d_x=d_x)
            x_init = simulator.sample()
            x_alt = simulator.sample()
            # ----------------- RANDOM FORESTS ---------------------
            z_opt, z_alt, x_r, x_a, t_r, t_a = run_sensitivity_experiment(
                simulator, x_init, x_alt, simulator.get_model, gurobiEnv,
                prescriptorType='rf', nbTrees=nbTrees)
            partialResultDf = pd.DataFrame(
                {'experiment': [experimentName],
                 'd_x': [d_x],
                 'nbSamples': [nbSamples],
                 'simIndex': [i],
                 'method': ['rf'],
                 'nbTrees': [nbTrees],
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
        resultDataframe.to_csv(
            folderName + '/' + experimentName
            + '_features_sens_n_'+str(nbSamples)+'_d_x_'+str(d_x)+'.csv')
