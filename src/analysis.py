"""
Functions to read, process, format, and export simulations results
to tables and figures.
"""
import numpy as np


def get_computation_times_table(df, experimentList, nList):
    table = init_table_list()
    # RF: T = 100
    dataDf = df[df['nbTrees'] == 100]
    table = fill_col_computation_times(table, dataDf, 2, nList, experimentList)
    # RF: T = 200
    dataDf = df[df['nbTrees'] == 200]
    table = fill_col_computation_times(table, dataDf, 3, nList, experimentList)
    # k-NN: k=10
    dataDf = df[df['nbNeighbors'] == 10]
    table = fill_col_computation_times(table, dataDf, 4, nList, experimentList)
    # k-NN: k=30
    dataDf = df[df['nbNeighbors'] == 30]
    table = fill_col_computation_times(table, dataDf, 5, nList, experimentList)
    return table


def get_small_computation_times_table(df, experimentList, nList):
    smallTable = init_small_table_list()
    # RF: T = 100
    dataDf = df[df['nbTrees'] == 100]
    smallTable = fill_col_computation_times(
        smallTable, dataDf, 2, nList, experimentList, startRow=2, increment=2)
    # k-NN: k=10
    dataDf = df[df['nbNeighbors'] == 10]
    smallTable = fill_col_computation_times(
        smallTable, dataDf, 3, nList, experimentList, startRow=2, increment=2)
    return smallTable


def get_percentage_features_changed(df, experimentList, nList):
    smallTable = init_small_table_list()
    # RF: T = 100
    dataDf = df[df['nbTrees'] == 100]
    smallTable = fill_col_percentage_features_changed(
        smallTable, dataDf, 2, nList, experimentList, startRow=2, increment=2)
    # k-NN: k=10
    dataDf = df[df['nbNeighbors'] == 10]
    smallTable = fill_col_percentage_features_changed(
        smallTable, dataDf, 3, nList, experimentList, startRow=2, increment=2)
    return smallTable


def get_symmetric_context_table(df, experimentList, nList):
    table = init_table_list()
    dataDf = df[df['nbTrees'] == 100]
    table = fill_col_symmetric_percentage(
        table, dataDf, 2, nList, experimentList)
    dataDf = df[df['nbTrees'] == 200]
    table = fill_col_symmetric_percentage(
        table, dataDf, 3, nList, experimentList)
    dataDf = df[df['nbNeighbors'] == 10]
    table = fill_col_symmetric_percentage(
        table, dataDf, 4, nList, experimentList)
    dataDf = df[df['nbNeighbors'] == 30]
    table = fill_col_symmetric_percentage(
        table, dataDf, 5, nList, experimentList)
    return table


def get_forest_depth_computation_times_table(df, experimentList, nList):
    table = init_forest_depth_table_list()
    col = 2
    depths = [3, 4, 5, 6]
    for depth in depths:
        dataDf = df[df['treeDepth'] == depth]
        table = fill_col_computation_times(table, dataDf, col,
                                           nList, experimentList, startRow=2)
        col = col+1
    return table


def get_distance_table(df, experimentList, nList):
    table = init_table_list()
    # RF: T = 100
    dataDf = df[df['nbTrees'] == 100]
    table = fill_col_distance(table, dataDf, 2, nList, experimentList)
    # RF: T = 200
    dataDf = df[df['nbTrees'] == 200]
    table = fill_col_distance(table, dataDf, 3, nList, experimentList)
    # k-NN: k=10
    dataDf = df[df['nbNeighbors'] == 10]
    table = fill_col_distance(table, dataDf, 4, nList, experimentList)
    # k-NN: k=30
    dataDf = df[df['nbNeighbors'] == 30]
    table = fill_col_distance(table, dataDf, 5, nList, experimentList)
    return table


def write_table_to_tex(table, fileName, pathToFiles):
    with open(pathToFiles+"/tables/"+fileName+".tex", "w") as f:
        f.write('\\begin{tabular}{*{10}{c}} \n')
        for row in table:
            f.write('    ')
            for c in range(len(row)):
                f.write(str(row[c]))
                if (c < len(row)-1):
                    f.write(' & ')
                else:
                    f.write(' \\\\ \n')
        f.write('\\bottomrule \n\\end{tabular}')


def init_small_table_list():
    return [
        [r'\toprule  ', '',
         r'\multicolumn{2}{c}{Relative explanation}',
         r'\multicolumn{2}{c}{Absolute explanation}'],
        [r'\cmidrule(lr){3-4} \cmidrule(lr){5-6}',
         'n', r'RF', r'k-NN', r'RF', r'k-NN'],
        [r'\midrule \multirow{3}{*}{NWS}', '$50$']+['']*4,
        ['', '$100$']+['']*4,
        ['', '$200$']+['']*4,
        [r'\cmidrule{2-6} \multirow{3}{*}{SHM}', '$50$']+['']*4,
        ['', '$100$']+['']*4,
        ['', '$200$']+['']*4,
        [r'\cmidrule{2-6} \multirow{3}{*}{SP}', '$50$']+['']*4,
        ['', '$100$']+['']*4,
        ['', '$200$']+['']*4,
        [r'\cmidrule{2-6} \multirow{3}{*}{c-SP}',
            '$50$']+['']*4,
        ['', '$100$']+['']*4,
        ['', '$200$']+['']*4]


def init_table_list():
    return [
        [r'\toprule  ', '',
         r'\multicolumn{4}{c}{Relative explanation}',
         r'\multicolumn{4}{c}{Absolute explanation}'],
        [r'\cmidrule(lr){3-6} \cmidrule(lr){7-10}',
         '', r'\multicolumn{2}{c}{RF}', r'\multicolumn{2}{c}{k-NN}',
         r'\multicolumn{2}{c}{RF}', r'\multicolumn{2}{c}{k-NN}'],
        [r'\cmidrule(lr){3-4} \cmidrule(lr){5-6}'
         + r' \cmidrule(lr){7-8} \cmidrule(lr){9-10}', '',
         '$T=100$', '$T=200$', '$k=10$', '$k=30$',
         '$T=100$', '$T=200$', '$k=10$', '$k=30$'],
        [r'\midrule \multirow{3}{*}{Newsvendor}', '$n=50$']+['']*8,
        ['', '$n=100$']+['']*8,
        ['', '$n=200$']+['']*8,
        [r'\cmidrule{2-10} \multirow{3}{*}{Shipment}', '$n=50$']+['']*8,
        ['', '$n=100$']+['']*8,
        ['', '$n=200$']+['']*8,
        [r'\cmidrule{2-10} \multirow{3}{*}{Shortest path}', '$n=50$']+['']*8,
        ['', '$n=100$']+['']*8,
        ['', '$n=200$']+['']*8,
        [r'\cmidrule{2-10} \multirow{3}{*}{\shortstack{CVaR\\ Shortest path}}',
            '$n=50$']+['']*8,
        ['', '$n=100$']+['']*8,
        ['', '$n=200$']+['']*8]


def init_forest_depth_table_list():
    return [
        [r'\toprule  ', '',
         r'\multicolumn{4}{c}{Relative explanation}',
         r'\multicolumn{4}{c}{Absolute explanation}'],
        [r'\cmidrule(lr){3-6} \cmidrule(lr){7-10}'
         + r' \multicolumn{2}{r}{Max. tree depth}',
         '$3$', '$4$', '$5$', '$6$',
         '$3$', '$4$', '$5$', '$6$'],
        [r'\midrule \multirow{3}{*}{Newsvendor}', '$n=50$']+['']*8,
        ['', '$n=100$']+['']*8,
        ['', '$n=200$']+['']*8,
        [r'\cmidrule{2-10} \multirow{3}{*}{Shipment}', '$n=50$']+['']*8,
        ['', '$n=100$']+['']*8,
        ['', '$n=200$']+['']*8,
        [r'\cmidrule{2-10} \multirow{3}{*}{Shortest path}', '$n=50$']+['']*8,
        ['', '$n=100$']+['']*8,
        ['', '$n=200$']+['']*8,
        [r'\cmidrule{2-10} \multirow{3}{*}{\shortstack{CVaR\\ Shortest path}}',
            '$n=50$']+['']*8,
        ['', '$n=100$']+['']*8,
        ['', '$n=200$']+['']*8]


def fill_col_computation_times(table, dataDf, colIndex,
                               nList, experimentList,
                               startRow=3, increment=4):
    row = startRow
    for experiment in experimentList:
        df1 = dataDf[dataDf['experiment'] == experiment]
        for n in nList:
            df2 = df1[df1['nbSamples'] == n]
            relTimes = np.array(df2['solvetime_relative'])
            absTimes = np.array(df2['solvetime_absolute'])
            table[row][colIndex] = "{:0.2f}".format(
                np.mean(relTimes[relTimes != 0.]))
            table[row][colIndex+increment] = "{:0.2f}".format(
                np.mean(absTimes[absTimes != 0.]))
            row = row + 1
    return table


def fill_col_percentage_features_changed(table, dataDf, colIndex,
                                         nList, experimentList,
                                         startRow=3, increment=4):
    row = startRow
    for experiment in experimentList:
        df1 = dataDf[dataDf['experiment'] == experiment]
        for n in nList:
            df2 = df1[df1['nbSamples'] == n]
            perc_change_rel = np.array(df2['perc_feat_change_rel'])
            perc_change_abs = np.array(df2['perc_feat_change_abs'])
            table[row][colIndex] = "{:0.2f}".format(
                np.mean(perc_change_rel[perc_change_rel != 0.0]))
            table[row][colIndex+increment] = "{:0.2f}".format(
                np.mean(perc_change_abs[perc_change_abs != 0.0]))
            row = row + 1
    return table


def fill_col_symmetric_percentage(table, dataDf, colIndex,
                                  nList, experimentList, startRow=3):
    row = startRow
    for experiment in experimentList:
        df1 = dataDf[dataDf['experiment'] == experiment]
        for n in nList:
            df2 = df1[df1['nbSamples'] == n]
            NB_REPETITIONS = len(df2)
            assert NB_REPETITIONS == 100
            relTimes = np.array(df2['solvetime_relative'])
            absTimes = np.array(df2['solvetime_absolute'])
            table[row][colIndex] = "{:0.0f}".format(
                100 * sum(relTimes == 0.) / NB_REPETITIONS)
            table[row][colIndex+4] = "{:0.0f}".format(
                100 * sum(absTimes == 0.) / NB_REPETITIONS)
            row = row + 1
    return table


def fill_col_distance(table, dataDf, colIndex,
                      nList, experimentList, startRow=3, increment=4):
    row = startRow
    for experiment in experimentList:
        df1 = dataDf[dataDf['experiment'] == experiment]
        for n in nList:
            df2 = df1[df1['nbSamples'] == n]
            relDistance = np.array(df2['distance_relative'])
            absDistance = np.array(df2['distance_absolute'])
            table[row][colIndex] = np.around(
                np.mean(relDistance[relDistance > 0.]), decimals=2)
            table[row][colIndex+increment] = np.around(
                np.mean(absDistance[absDistance > 0.]), decimals=2)
            row = row + 1
    return table


def get_x_as_list(xTypeString, df):
    x = []
    for i in range(len(df.index)):
        x_string = df[xTypeString].values[i].strip('[]')
        x.append(np.fromstring(x_string, sep=' '))
    return x


def percentage_relevant_feature_change(
        spuriousDf, x_init, x_rel, x_abs):
    dist_rel = []
    dist_abs = []
    dist_rel_information_only = []
    dist_abs_information_only = []
    n_rows = len(spuriousDf.index)
    # Measure counterfactual distance
    for i in range(n_rows):
        # Relative explanation distance
        if len(x_rel[i]) == 0:
            dist_rel.append(-1)
        else:
            dist_rel.append(np.linalg.norm(x_init[i] - x_rel[i], ord=1))
        # Hard distance
        if len(x_abs[i]) == 0:
            dist_abs.append(-1)
        else:
            dist_abs.append(np.linalg.norm(x_init[i] - x_abs[i], ord=1))
        # Relative explanation distance: informative features only
        if len(x_rel[i]) == 0:
            dist_rel_information_only.append(-1)
        else:
            dist_rel_information_only.append(
                np.linalg.norm(x_init[i][0:2] - x_rel[i][0:2], ord=1))
        # Hard distance: informative features only
        if len(x_abs[i]) == 0:
            dist_abs_information_only.append(-1)
        else:
            dist_abs_information_only.append(
                np.linalg.norm(x_init[i][0:2] - x_abs[i][0:2], ord=1))
    # Store in result df
    relPercentage = np.array(dist_rel_information_only) / \
        np.array(dist_rel) * 100
    absPercentage = np.array(dist_abs_information_only) / \
        np.array(dist_abs) * 100
    # Store in result df
    spuriousDf['relative_percentage'] = relPercentage
    spuriousDf['absolute_percentage'] = absPercentage
    return spuriousDf


def re_encode_categorical_features(x, oneHotEncodedFeatures):
    # Encode the value of categorical features to new columns
    for f in oneHotEncodedFeatures:
        colIndices = oneHotEncodedFeatures[f]
        nbPossibleValues = len(colIndices)
        for i in range(nbPossibleValues):
            if x[colIndices[i]] == 1.0:
                # Add a feature at the end
                x = np.r_[x, i / (nbPossibleValues-1)]
    # Delete all columns corresponding to one-hot encodded features
    colsToDelete = list(oneHotEncodedFeatures.values())
    colsToDelete = [item for sublist in colsToDelete for item in sublist]
    x = np.delete(x, colsToDelete, axis=0)
    return x
