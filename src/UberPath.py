"""
Model and solve a CVaR Shortest Path problem with weighted SAA.
"""
import os as os
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import geopandas as gpd


class UberPath:
    """
    Model and solve a CVaR Shortest Path problem.
    """

    def __init__(self, Y_train, A_mat, b_vec,
                 weights=None, alpha=0.7, verbose=False):
        """ Initialize model """
        self.L = Y_train.shape[1]
        self.n = Y_train.shape[0]
        self.Y_train = Y_train
        self.A_mat = A_mat
        self.b_vec = b_vec
        self.riskType = 'CVaR'
        # Define adaptive weights
        if weights is None:
            self.weights = np.ones(self.n)/self.n
        else:
            self.weights = weights
            assert len(weights) == self.n
        self.alpha = alpha
        self.verbose = verbose

    def build(self):
        """
        Sets up a shortest path decision problem over
        Los Angeles downtown area.
        """
        self.model = gp.Model('Cvar_shortest_path')
        # Define flow variables and constraints
        self.zVar = pd.Series(self.model.addVars(
            self.L, vtype=GRB.BINARY, name='z'), index=range(self.L))
        # Flow constraints
        LP_constraints = []
        for i in range(self.A_mat.shape[0]):
            LP_constraints.append(self.model.addConstr(
                self.A_mat[i, :].dot(self.zVar) == self.b_vec[i]))
        # - CVaR -
        # Auxiliary variables for CVaR calculation
        self.auxVar = self.model.addVar(lb=-float('inf'), name='auxiliary')
        maxCost = pd.Series(self.model.addVars(self.n, lb=0, name='maxCost'),
                            index=range(self.n))
        max_constraints = []
        for s in range(self.n):
            sampleCost = np.sum(
                [self.Y_train[s, e] * self.zVar[e] for e in range(self.L)])
            max_constraints.append(
                self.model.addConstr(maxCost[s] >= sampleCost - self.auxVar))
        risk = (1/(1 - self.alpha)) * \
            np.sum(np.dot(self.weights, maxCost)) + self.auxVar
        self.model.setObjective(risk, GRB.MINIMIZE)
        if not self.verbose:
            self.model.setParam('OutputFlag', 0)

    def solve(self):
        self.model.optimize()
        zOpt = dict()
        for ind in range(self.L):
            zOpt[ind] = self.zVar[ind].X
            # Post-process numerical instabilities
            if -1e-6 < zOpt[ind] < 1e-6:
                zOpt[ind] = 0
            elif (1+1e-6) > zOpt[ind] > (1-1e-6):
                zOpt[ind] = 1
            else:
                print('Warning: numerical instability at edge {}'.format(
                    ind))
                print('Gurobi found value: ', zOpt[ind])
        return zOpt

    def costFunction(self, z, y, s=None):
        assert len(z) == len(y)
        assert len(z) == 93
        return sum(np.array(list(z.values())) * y)

    def riskFunctional(self, z, y, weights):
        n = y.shape[0]
        indices = range(n)
        # Calculate cost in each historical sample
        costVector = np.asarray([self.costFunction(z, y[i])
                                for i in indices], dtype=np.float32)
        # Sort the weights
        sortedIndices = np.argsort(-costVector)
        cumulativeWeights = np.cumsum(weights[sortedIndices])
        # Get index of VaR sample and determine its residual
        VaRindex = next(i for i, v in enumerate(
            cumulativeWeights) if v >= (1-self.alpha))
        residualWeight = (1-self.alpha) - cumulativeWeights[VaRindex-1]
        try:
            if VaRindex > 0:
                assert (residualWeight+1e-12) >= 0
        except AssertionError:
            print('! Error calculating CVaR !')
            print('Residual weight of VaR sample is negative: ',
                  residualWeight)
            raise
        # Calculate CVaR
        CVaRresidual = residualWeight * costVector[sortedIndices[VaRindex]]
        CVaR = 1/(1-self.alpha) * (sum(
            [weights[sortedIndices[i]] * costVector[sortedIndices[i]]
             for i in range(VaRindex)]) + CVaRresidual)
        return CVaR


# Define functions used in Newsvendor experiments
def get_node_list():
    nodeList = [
        "1221", "1222", "1220", "1230", "1223", "1224", "1390", "1229", "1228",
        "1234", "1380", "1232", "1233", "1235", "1254", "1255", "1263", "1382",
        "1237", "1252", "1251", "1236", "1253", "1239", "1238", "1250", "1249",
        "1248", "1258", "1257", "1260", "1262", "1384", "1240", "1241", "1246",
        "1247", "1256", "1259", "1261", "1243", "1245", "1242", "1383", "1244"]
    return nodeList


def get_census_dict():
    census = dict()
    census["1221"] = '206032'
    census["1222"] = '206050'
    census["1220"] = '206031'
    census["1230"] = '207400'
    census["1223"] = '206200'
    census["1224"] = '206300'
    census["1390"] = '226002'
    census["1229"] = '207302'
    census["1228"] = '207301'
    census["1234"] = '207900'
    census["1380"] = '224010'
    census["1232"] = '207502'
    census["1233"] = '207710'
    census["1235"] = '208000'
    census["1254"] = '209200'
    census["1255"] = '209300'
    census["1263"] = '210010'
    census["1382"] = '224200'
    census["1237"] = '208302'
    census["1252"] = '209103'
    census["1251"] = '209102'
    census["1236"] = '208301'
    census["1253"] = '209104'
    census["1239"] = '208402'
    census["1238"] = '208401'
    census["1250"] = '208904'
    census["1249"] = '208903'
    census["1248"] = '208902'
    census["1258"] = '209403'
    census["1257"] = '209402'
    census["1260"] = '209520'
    census["1262"] = '209820'
    census["1384"] = '224320'
    census["1240"] = '208501'
    census["1241"] = '208502'
    census["1246"] = '208801'
    census["1247"] = '208802'
    census["1256"] = '209401'
    census["1259"] = '209510'
    census["1261"] = '209810'
    census["1243"] = '208620'
    census["1245"] = '208720'
    census["1242"] = '208610'
    census["1383"] = '224310'
    census["1244"] = '208710'
    return census


def get_gps_coordinates(dirPath, census):
    """Read and store GPS coordinates of census tract centroid."""
    # Load shapefile data
    shapefile = gpd.read_file(os.path.join(dirPath, "tl_2018_06_tract.shp"))
    # Read GPS coordinates
    centroidGps = dict()
    for censusTrack in census.values():
        cenString = censusTrack[:-2]+'.'+censusTrack[-2:]
        if cenString[-2:] == '00':
            cenString = cenString[:-3]
        filteredShp = shapefile[shapefile['NAMELSAD'].str.contains(cenString)]
        if len(filteredShp) > 0:
            centroidGps[censusTrack] = [
                float(np.array(filteredShp['INTPTLON'])[0]),
                float(np.array(filteredShp['INTPTLAT'])[0])]
        else:
            print('No GPS coordinate for Census tract: ', censusTrack)
    return centroidGps
