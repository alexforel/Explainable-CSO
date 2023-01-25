"""
Analyze the results of the repetated experiments:
    - read simulations results from csv files,
    - calculate kpis,
    - plot relevant figures,
    - export results to LaTeX tables, images, and csv files.
"""
# Import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import local functions
from src.analysis import write_table_to_tex
from src.analysis import get_computation_times_table
from src.analysis import get_small_computation_times_table
from src.analysis import get_forest_depth_computation_times_table
from src.analysis import get_distance_table
from src.analysis import get_x_as_list
from src.analysis import percentage_relevant_feature_change
from src.analysis import get_symmetric_context_table

# Set global plotting style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['lines.linewidth'] = 4
plt.rcParams["figure.figsize"] = (18, 6)
plt.rcParams.update({"axes.grid": True,
                     "grid.color": "grey"})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
sns.set_palette('colorblind')
sns.set_context('paper')
SUBPLT_SIZE = (8, 2)

# ------ Read simulations results ------
# Import result files and concatenate them into a single Dataframe
experimentList = ['newsvendor', 'shipment', 'path', 'cvar-path']
nList = [50, 100, 200]
dirname = os.path.dirname(__file__)
outputPath = os.path.join(dirname, "output",)
filepaths = [outputPath+"/"
             + f for f in os.listdir(outputPath) if f.endswith('.csv')]

# -------- Computation times --------
# -> Table 1 and 5 in paper
# Read and concatenate results into one single dataframe
resultsFiles = [p for p in filepaths if 'synth_results' in p]
resultDf = pd.concat(map(pd.read_csv, resultsFiles))
# -- Long table used in appendix --
# Create and formate table
compTimesTable = get_computation_times_table(
    resultDf, experimentList, nList)
# Write and export table to tex file
write_table_to_tex(compTimesTable, 'computation_times', outputPath)
# -- Short table used in paper body --
smallCompTimesTable = get_small_computation_times_table(
    resultDf, experimentList, nList)
# Write and export table to tex file
write_table_to_tex(smallCompTimesTable, 'small_computation_times', outputPath)

# -------- Symmetric contexts --------
# -> Table 4 in paper
symmTable = get_symmetric_context_table(resultDf, experimentList, nList)
write_table_to_tex(symmTable, 'percentage_symmetric', outputPath)

# -------- Counterfactual distance --------
# -> Table 7 in paper
# Read list of explanations
x_init = get_x_as_list('x_init', resultDf)
x_rel = get_x_as_list('x_rel', resultDf)
x_abs = get_x_as_list('x_abs', resultDf)
# Add column to dataframe with distance
dist_rel = []
dist_abs = []
for i in range(len(resultDf.index)):
    if len(x_rel[i]) == 0:
        dist_rel.append(-1)
    else:
        dist_rel.append((1/len(x_init[i]))
                        * np.linalg.norm(x_init[i] - x_rel[i], ord=1))
    if len(x_abs[i]) == 0:
        dist_abs.append(-1)
    else:
        dist_abs.append((1/len(x_init[i]))
                        * np.linalg.norm(x_init[i] - x_abs[i], ord=1))
# Store in result df
resultDf['distance_relative'] = dist_rel
resultDf['distance_absolute'] = dist_abs
# Create and formate table
distanceTable = get_distance_table(resultDf, experimentList, nList)
# Write and export table to tex file
write_table_to_tex(distanceTable, 'explanation_distance', outputPath)

# -- Direction of explanations --
# -> Figure 6
# Measure the correlation between the explanations
# and the alternative contexts
directionDf = resultDf[(resultDf['nbTrees'] == 100)
                       + (resultDf['nbNeighbors'] == 10)]
directionDf = directionDf[directionDf['experiment'] == 'newsvendor']
corr_rel = []
corr_abs = []
n_rows = len(directionDf.index)
x_init = get_x_as_list('x_init', directionDf)
x_rel = get_x_as_list('x_rel', directionDf)
x_abs = get_x_as_list('x_abs', directionDf)
x_alt = get_x_as_list('x_alt', directionDf)
# Measure counterfactual change
for i in range(n_rows):
    # Calculate direction of explanations
    true_direction = x_alt[i] - x_init[i]
    rel_direction = x_rel[i] - x_init[i]
    # Normalize directions
    true_direction = true_direction / np.linalg.norm(true_direction)
    rel_direction = rel_direction / np.linalg.norm(rel_direction)
    # Calculate dot product and append to lists
    corr_rel.append(np.dot(true_direction, rel_direction))
    if len(x_abs[i]) == 0:
        corr_abs.append(None)
    else:
        abs_direction = x_abs[i] - x_init[i]
        abs_direction = abs_direction / np.linalg.norm(abs_direction)
        corr_abs.append(np.dot(true_direction, abs_direction))
directionDf['Relative explanations'] = corr_rel
directionDf['Absolute explanations'] = corr_abs
directionDf = directionDf.rename(
    columns={'corr_rel': 'Relative explanations'})
directionDf = directionDf.rename(
    columns={'corr_abs': 'Absolute explanations'})
# Rename fields for plotting
directionDf.loc[directionDf.experiment
                == 'newsvendor', 'experiment'] = 'Newsvendor'
directionDf.loc[directionDf.method == 'rf', 'method'] = 'RF'
directionDf.loc[directionDf.method == 'knn', 'method'] = 'k-NN'
explanationList = ['Relative explanations', 'Absolute explanations']

PRT_SIZE = (4.5, 2.5)
yMins = [-1, 0.0]
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=PRT_SIZE)
for i in range(len(explanationList)):
    sns.boxplot(ax=axs[i], x="nbSamples", y=explanationList[i],
                hue="method", data=directionDf, showmeans=True,
                linewidth=2,
                meanprops={"marker": "o", "markerfacecolor": "white",
                           "markeredgecolor": "black",
                           "markersize": "7"})
    axs[i].set_title(explanationList[i])
    handles, labels = axs[i].get_legend_handles_labels()
    axs[i].legend(handles=handles[0:], labels=labels[0:], loc='lower right')
    axs[i].set_ylim(ymin=yMins[i], ymax=1.0)
    axs[i].set_ylabel(r"Correlation")
    axs[i].grid(visible=True, which='both', color='gray', linewidth=0.1)
    axs[i].set_axisbelow(True)
axs[0].get_legend().remove()
axs[0].set_xlabel(None)
axs[1].set_xlabel(r"Sample size $n$")
fig.tight_layout()
plt.savefig(outputPath+'/figs/newsvendor_corr.pdf', bbox_inches='tight')

# ----- Problem complexity -----
# -> Figure 3 and 4
# Read and concatenate results into one single dataframe
complexFiles = [p for p in filepaths if 'complex' in p]
complexDf = pd.concat(map(pd.read_csv, complexFiles))
# Replace 0.0 by NaN
complexDf[['solvetime_relative', 'solvetime_absolute']] = complexDf[[
    'solvetime_relative', 'solvetime_absolute']].replace(0, np.nan)
# Transform runtime to log seconds
complexDf['solvetime_relative'] = np.log10(complexDf['solvetime_relative'])
complexDf['solvetime_absolute'] = np.log10(complexDf['solvetime_absolute'])
# Rename columns and fields
complexDf = complexDf.rename(columns={'solvetime_relative': 'Relative time'})
complexDf = complexDf.rename(columns={'solvetime_absolute': 'Absolute time'})
complexDf.loc[complexDf.experiment
              == 'newsvendor', 'experiment'] = 'Newsvendor'
complexDf.loc[complexDf.experiment == 'path', 'experiment'] = 'Shortest path'
complexDf.loc[complexDf.method == 'rf', 'method'] = 'RF'
complexDf.loc[complexDf.method == 'knn', 'method'] = 'k-NN'

PRT_SIZE2 = (4.5, 3)
explanationList = ['Relative', 'Absolute']
problemList = ['Newsvendor', 'Shortest path']
for j in range(len(problemList)):
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=PRT_SIZE2)
    tempDf = complexDf[complexDf['experiment'] == problemList[j]]
    for i in range(len(explanationList)):
        sns.boxplot(ax=axs[i], x="problemSens", y=explanationList[i]+" time",
                    hue="method", data=tempDf, showmeans=True,
                    linewidth=2,
                    meanprops={"marker": "o", "markerfacecolor": "white",
                               "markeredgecolor": "black",
                               "markersize": "7"})
        axs[i].set_title(explanationList[i]+' explanations')
        handles, labels = axs[i].get_legend_handles_labels()
        axs[i].legend(handles=handles[0:], labels=labels[0:],
                      ncol=2, loc='upper left')
        axs[i].set_ylabel(r"Time [in $s$]")
        axs[i].set_yticks(np.arange(-1, 4))
        axs[i].set_yticklabels(10.0**np.arange(-1, 4))
        minor_yticks = np.log10(
            np.concatenate((np.arange(1, 10) * 1,
                            np.arange(1, 10) * 0.1,
                            np.arange(1, 10) * 10,
                            np.arange(1, 10) * 100)).astype(float))
        axs[i].set_yticks(minor_yticks, minor=True)
        axs[i].grid(visible=True, which='major', color='gray', linewidth=0.1)
        axs[i].set_axisbelow(True)
        if j == 0:
            axs[i].set_ylim(ymax=1.5)
        else:
            axs[i].set_ylim(ymax=3.2)
    axs[1].get_legend().remove()
    axs[0].set_xlabel(None)
    axs[1].set_xlabel(r"Problem complexity")
    fig.tight_layout()
    plt.savefig(outputPath+'/figs/'+problemList[j]
                + '_problem_cplx.pdf', bbox_inches='tight')

# ----- Spurious features -----
# -> Figure 9 and 10
spuriousFiles = [p for p in filepaths if 'spurious' in p]
spuriousDf = pd.concat((pd.read_csv(f) for f in spuriousFiles),
                       ignore_index=True)
nList = [50, 100, 200, 400, 600, 800]
NB_ROWS = 100 * len(nList) * 1  # * forest size variations
assert len(spuriousDf.index) == NB_ROWS
# Read init points and explanations
x_init = get_x_as_list('x_init', spuriousDf)
x_rel = get_x_as_list('x_rel', spuriousDf)
x_abs = get_x_as_list('x_abs', spuriousDf)
assert len(x_init) == NB_ROWS
assert len(x_rel) == NB_ROWS
assert len(x_abs) == NB_ROWS
# Calculate percentage of relevant feature change and add to df
spuriousDf = percentage_relevant_feature_change(
    spuriousDf, x_init, x_rel, x_abs)
# Rename columns and fields
spuriousDf = spuriousDf.rename(
    columns={'relative_percentage': 'Relative explanations'})
spuriousDf = spuriousDf.rename(
    columns={'absolute_percentage': 'Absolute explanations'})

explanationList = ['Relative explanations', 'Absolute explanations']
fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=SUBPLT_SIZE)
for i in range(len(explanationList)):
    sns.violinplot(ax=axs[i], x="nbSamples", y=explanationList[i],
                   hue='method',
                   cut=0, data=spuriousDf, showmeans=True,
                   linewidth=2)
    axs[i].set_title(explanationList[i])
    axs[i].set_ylabel(None)
    fig.supylabel(r"Relevant change [in\%]")
    axs[i].set_xlabel(r"Sample size $n$")
    axs[i].grid(visible=True, which='both', color='gray', linewidth=0.1)
    axs[i].set_axisbelow(True)
    axs[i].get_legend().remove()
    axs[i].set_ylim(ymin=0.0, ymax=100)
fig.tight_layout()
plt.savefig(outputPath+'/figs/spurious_perc.pdf', bbox_inches='tight')

# -- Direction of explanations --
corr_rel = []
corr_abs = []
n_rows = len(spuriousDf.index)
x_alt = get_x_as_list('x_alt', spuriousDf)
# Measure counterfactual change
for i in range(n_rows):
    # Calculate direction of explanations
    true_direction = x_alt[i] - x_init[i]
    rel_direction = x_rel[i] - x_init[i]
    # Normalize directions
    true_direction = true_direction / np.linalg.norm(true_direction)
    rel_direction = rel_direction / np.linalg.norm(rel_direction)
    # Calculate dot product and append to lists
    corr_rel.append(np.dot(true_direction, rel_direction))
    if len(x_abs[i]) == 0:
        corr_abs.append(None)
    else:
        abs_direction = x_abs[i] - x_init[i]
        abs_direction = abs_direction / np.linalg.norm(abs_direction)
        corr_abs.append(np.dot(true_direction, abs_direction))
spuriousDf['Relative correlation'] = corr_rel
spuriousDf['Absolute correlation'] = corr_abs

explanationList = ['Relative correlation', 'Absolute correlation']
fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=SUBPLT_SIZE)
for i in range(len(explanationList)):
    sns.violinplot(ax=axs[i], x="nbSamples", y=explanationList[i],
                   hue='method',
                   cut=0, data=spuriousDf, showmeans=True,
                   linewidth=2)
    axs[i].set_title(explanationList[i])
    axs[i].set_ylabel(None)
    fig.supylabel(r"Correlation")
    axs[i].set_xlabel(r"Sample size $n$")
    axs[i].grid(visible=True, which='both', color='gray', linewidth=0.1)
    axs[i].set_axisbelow(True)
    axs[i].set_ylim(ymin=-1.0, ymax=1.0)
    axs[i].get_legend().remove()
fig.tight_layout()
plt.savefig(outputPath+'/figs/spurious_corr.pdf', bbox_inches='tight')

# ----- Sensitivity to forest depth -----
# -> Table 6
nList = [50, 100, 200]
# Read and concatenate results into one single dataframe
depthPath = outputPath+"/forest_depth/"
filepaths = [depthPath+"/"
             + f for f in os.listdir(depthPath) if f.endswith('.csv')]
resultsFiles = [p for p in filepaths if 'synth_results' in p]
resultDf = pd.concat(map(pd.read_csv, resultsFiles))
# Create and formate table
compTimesTable = get_forest_depth_computation_times_table(
    resultDf, experimentList, nList)
# Write and export table to tex file
write_table_to_tex(compTimesTable, 'depth_comp_times', outputPath)

# ----- Sensitivity to number of features and sample size -----
# -> Export csv data used to create Figure 2 in tikz
# Read and concatenate results into one single dataframe
featPath = outputPath+"/path_feat/"
filepaths = [featPath + f for f in os.listdir(featPath) if f.endswith('.csv')]
featuresFiles = [p for p in filepaths if 'feat' in p]
featuresDf = pd.concat((pd.read_csv(f) for f in featuresFiles),
                       ignore_index=True)
nList = [100, 500, 1000, 5000]
d_xList = np.sort(featuresDf['d_x'].unique())
# Get average computations times and std
avgRelTime = np.zeros((len(nList), len(d_xList)))
avgAbsTime = np.zeros((len(nList), len(d_xList)))
stdRelTime = np.zeros((len(nList), len(d_xList)))
stdAbsTime = np.zeros((len(nList), len(d_xList)))
for i in range(len(nList)):
    tempDf = featuresDf[featuresDf['nbSamples'] == nList[i]]
    for j in range(len(d_xList)):
        tempDf2 = tempDf[tempDf['d_x'] == d_xList[j]]
        relTimes = np.array(tempDf2['solvetime_relative'])
        absTimes = np.array(tempDf2['solvetime_absolute'])
        avgRelTime[i, j] = np.mean(relTimes[relTimes != 0.])
        avgAbsTime[i, j] = np.mean(absTimes[absTimes != 0.])
        stdRelTime[i, j] = np.std(relTimes[relTimes != 0.])
        stdAbsTime[i, j] = np.std(absTimes[absTimes != 0.])
# Export to csv file
dfToExport = pd.DataFrame({'d_x': d_xList})
for i, n in enumerate(nList):
    dfToExport[str(n)+'_avgRelTime'] = avgRelTime[i, :]
    dfToExport[str(n)+'_avgRelTime_plus_std'] = (
        avgRelTime[i, :]+stdRelTime[i, :])
    dfToExport[str(n)+'_avgRelTime_minus_std'] = (
        avgRelTime[i, :]-stdRelTime[i, :])
    dfToExport[str(n)+'_avgBsTime'] = avgAbsTime[i, :]
    dfToExport[str(n)+'_avgBsTime_plus_std'] = (
        avgAbsTime[i, :]+stdAbsTime[i, :])
    dfToExport[str(n)+'_avgBsTime_minus_std'] = (
        avgAbsTime[i, :]-stdAbsTime[i, :])
folderName = os.path.join(outputPath, "csv")
os.makedirs(folderName, exist_ok=True)
dfToExport.to_csv(folderName+'/path_feat_sens.csv', index=False)

# ----- Value of dual reformulation for linear decision problems -----
# -> Figure 8
# Read and concatenate results into one single dataframe
filepaths = [outputPath+"/"
             + f for f in os.listdir(outputPath) if f.endswith('.csv')]
resultsFiles = [p for p in filepaths if 'dual' in p]
resultDf = pd.concat(map(pd.read_csv, resultsFiles))
# Rename methods
methodList = ['Random forest prescriptors', 'Nearest-neighbor prescriptors']
resultDf.loc[resultDf.method == 'rf', 'method'] = methodList[0]
resultDf.loc[resultDf.method == 'knn', 'method'] = methodList[1]
# Remove symmetric contexts
resultDf = resultDf.drop(resultDf[resultDf['solvetime_absolute'] <= 0.].index)

# Extract dataframes with and without dual formulation
dualDf = resultDf[resultDf.useDual]
iterativeDf = resultDf[resultDf.useDual == False]
# Calculate relative times
dualTimes = np.array(dualDf.solvetime_absolute)
iterativeTimes = np.array(iterativeDf.solvetime_absolute)
iterativeDf['relativeTimes'] = np.log10(iterativeTimes / dualTimes)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=SUBPLT_SIZE)
for i in range(len(methodList)):
    tempDf = iterativeDf[iterativeDf['method'] == methodList[i]]
    sns.boxplot(ax=axs[i], x="nbSamples", y='relativeTimes',
                hue='useDual', data=tempDf, showmeans=True,
                linewidth=2,
                meanprops={"marker": "o", "markerfacecolor": "white",
                           "markeredgecolor": "black",
                           "markersize": "7"})
    axs[i].set_title(methodList[i])
    axs[i].set_ylabel(None)
    fig.supylabel(r"Time ratio")
    axs[i].set_xlabel(r"Sample size $n$")
    axs[i].grid(visible=True, which='both', color='gray', linewidth=0.1)
    axs[i].set_axisbelow(True)
    axs[i].get_legend().remove()
    # Set y-axis to log scale
    axs[i].set_yticks(np.arange(-1, 2))
    axs[i].set_yticklabels(10.0**np.arange(-1, 2))
    minor_yticks = np.log10(
        np.concatenate((np.arange(1, 10) * 1,
                        np.arange(1, 10) * 0.1)).astype(np.float))
fig.tight_layout()
plt.savefig(outputPath+'/figs/dual_vs_iterative.pdf', bbox_inches='tight')
