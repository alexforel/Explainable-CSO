"""
Model and solve a shipment planning problem:
        simulation setting and functions are adapted from
            Bertsimas, D., & Kallus, N. (2020). From predictive to
            prescriptive analytics. Management Science, 66(3), 1025-1044.
"""
import numpy as np
from scipy.spatial import distance_matrix
import gurobipy as gp
from gurobipy import GRB


class Shipment:
    """
    Model and solve a shipment planning problem.
    """

    def __init__(self, demands, d_y, gurobiEnv,
                 weights=None, verbose=False):
        """ Initialize model """
        self.nbSamples = len(demands)
        self.verbose = verbose
        self.demands = demands
        self.d_y = d_y  # number of locations
        self.d_z = 4  # number of warehouses
        self.shipCost = 10   # in $ per distance unit
        self.prodCost = 5    # $ per unit
        self.spotCost = 100  # $ per unit
        # Calculate distance matrix:
        #   "We take locations spaced evenly on the 2-dimensional
        #    unit circle and warehouses spaced evenly on the circle
        #    of radius 0.85. "
        warehouses = [[0.85, 0], [0., 0.85], [-0.85, 0], [0, -0.85]]
        self.distanceMatrix = np.zeros((self.d_z, self.d_y))
        for i in range(self.d_y):
            self.distanceMatrix[:, i] = distance_matrix(
                    [[np.cos(i * np.pi/(self.d_y/2)),
                      np.sin(i * np.pi/(self.d_y/2))]],
                    warehouses)
        self.gurobiEnv = gurobiEnv
        self.riskType = 'Expectation'
        # Define adaptive weights
        if weights is None:
            self.weights = np.ones(self.nbSamples)/self.nbSamples
        else:
            self.weights = weights
            assert len(weights) == self.nbSamples

    def _array_recourse_decision(self, spot, ship):
        spotArray = np.zeros((self.d_z, self.nbSamples))
        shipArray = np.zeros((self.d_z, self.d_y, self.nbSamples))
        for i in range(self.d_z):
            for s in range(self.nbSamples):
                spotArray[i, s] = spot[i, s]
                for j in range(self.d_y):
                    shipArray[i, j, s] = ship[i, j, s]
        return spotArray, shipArray

    def build(self):
        self.model = gp.Model("shipment", env=self.gurobiEnv)
        self.model.setParam('OutputFlag', int(self.verbose))
        # ---- Init decision variables ----
        # First-stage decision: produce at warehouses
        self.production = dict()
        for i in range(self.d_z):
            self.production[i] = self.model.addVar(
                lb=0, vtype=GRB.CONTINUOUS, name="prod_i"+str(i))
        # Second-stage variables: shipments and spot orders
        self.shipment = dict()
        for i in range(self.d_z):
            for j in range(self.d_y):
                for s in range(self.nbSamples):
                    self.shipment[i, j, s] = self.model.addVar(
                        lb=0, vtype=GRB.CONTINUOUS,
                        name="ship_ij"+str(i)+str(j))
        self.spotOrders = dict()
        for i in range(self.d_z):
            for s in range(self.nbSamples):
                self.spotOrders[i, s] = self.model.addVar(
                    lb=0, vtype=GRB.CONTINUOUS, name="spot_i"+str(i))
        # ---- Define model constraints ----
        for s in range(self.nbSamples):
            # Satisfy demand with shipments
            for j in range(self.d_y):
                self.model.addConstr(
                        sum([self.shipment[i, j, s] for i in range(self.d_z)])
                        >= self.demands[s, j])
            # Flow conservation: production and spot orders cover shipments
            for i in range(self.d_z):
                self.model.addConstr(
                        sum([self.shipment[i, j, s] for j in range(self.d_y)])
                        <= self.production[i] + self.spotOrders[i, s])
        # ---- Define objective function ----
        firstStageCosts = sum([self.prodCost * self.production[i]
                              for i in range(self.d_z)])
        secondStageCosts = gp.LinExpr(0.0)
        for s in range(self.nbSamples):
            spotCosts = sum([self.spotCost * self.spotOrders[i, s]
                            for i in range(self.d_z)])
            shipmentCosts = sum(
                [sum([self.shipCost * self.distanceMatrix[i, j]
                      * self.shipment[i, j, s] for j in range(self.d_y)])
                 for i in range(self.d_z)])
            secondStageCosts += self.weights[s] * (spotCosts + shipmentCosts)
        self.obj = firstStageCosts + secondStageCosts
        self.model.setObjective(self.obj, GRB.MINIMIZE)

    def solve(self):
        self.model.optimize()
        prodOpt = dict()
        spotOpt = dict()
        shipOpt = dict()
        for i in range(self.d_z):
            prodOpt[i] = self.production[i].X
            for s in range(self.nbSamples):
                spotOpt[i, s] = self.spotOrders[i, s].X
                for j in range(self.d_y):
                    shipOpt[i, j, s] = self.shipment[i, j, s].X
        return prodOpt, spotOpt, shipOpt

    def costFunction(self, z, y, s=None):
        if s is None:
            prod, spot, ship = z
        else:
            prod, spot, ship = z
            spotArray, shipArray = self._array_recourse_decision(spot, ship)
            spot = spotArray[:, s]
            ship = shipArray[:, :, s]
        prodCosts = sum([self.prodCost * prod[i]
                         for i in range(self.d_z)])
        spotCosts = sum([self.spotCost * spot[i]
                        for i in range(self.d_z)])
        shipmentCosts = sum(
            [sum([self.shipCost * self.distanceMatrix[i, j]
                  * ship[i, j] for j in range(self.d_y)])
             for i in range(self.d_z)])
        return prodCosts + spotCosts + shipmentCosts

    def riskFunctional(self, z, demand, weights):
        """ Expected value """
        indices = range(len(demand))
        prod, spot, ship = z
        spotArray, shipArray = self._array_recourse_decision(spot, ship)
        expectedCost = np.sum(
            [self.costFunction((prod, spotArray[:, s], shipArray[:, :, s]),
                               demand[s]) * weights[s] for s in indices])
        return expectedCost
