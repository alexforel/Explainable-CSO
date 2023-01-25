"""
Model and solve a muli-item newsvendor with budget.
"""
import numpy as np
from scipy.stats import truncnorm
import gurobipy as gp
from gurobipy import GRB


class Newsvendor:
    """
    Model and solve a muli-item newsvendor with budget.
    """

    def __init__(self, demands, nbProducts, gurobiEnv,
                 weights=None, verbose=False):
        """ Initialize model """
        self.nbSamples = len(demands)
        self.nbProducts = nbProducts
        self.demands = demands
        # Define problem parameters
        self.budget = 5 * nbProducts
        self.overCosts = 1+np.array(range(self.nbProducts))
        self.underCosts = 10 * self.overCosts
        # Define adaptive weights
        if weights is None:
            # Non-contextual SAA
            self.weights = np.ones(self.nbSamples)/self.nbSamples
        else:
            self.weights = weights
            assert len(weights) == self.nbSamples
        # Define optimization type and parameters
        self.verbose = verbose
        self.gurobiEnv = gurobiEnv
        self.riskType = 'Expectation'

    def build(self):
        self.model = gp.Model("news", env=self.gurobiEnv)
        if not self.verbose:
            self.model.setParam('OutputFlag', 0)
        # ---- Init decision variables ----
        # Main decision variable: order quantity
        self.quantity = dict()
        # Auxiliary variables to track over/underage costs
        self.underageCosts = dict()
        self.overageCosts = dict()
        # Auxiliary variables to track over/underage volumes
        # in each scenario
        self.underVol = dict()
        self.overVol = dict()
        for p in range(self.nbProducts):
            self.quantity[p] = self.model.addVar(
                lb=0, ub=self.budget, vtype=GRB.CONTINUOUS,
                name="order_quantity_p"+str(p))
            self.underageCosts[p] = self.model.addVar(
                lb=0, vtype=GRB.CONTINUOUS, name="under_costs_p"+str(p))
            self.overageCosts[p] = self.model.addVar(
                lb=0, vtype=GRB.CONTINUOUS, name="over_costs_p"+str(p))
            self.underVol[p] = dict()
            self.overVol[p] = dict()
            for s in range(self.nbSamples):
                self.underVol[p][s] = self.model.addVar(
                    lb=0, vtype=GRB.CONTINUOUS,
                    name="under_p" + str(p) + "_scen"+str(s))
                self.overVol[p][s] = self.model.addVar(
                    lb=0, vtype=GRB.CONTINUOUS,
                    name="over_p" + str(p) + "_scen"+str(s))
        # ---- Define model constraints ----
        # Capacity constraint
        self.model.addConstr(
            gp.quicksum(self.quantity[p] for p in range(self.nbProducts))
            <= self.budget)
        for p in range(self.nbProducts):
            # Over and under volume in each scenario
            for s in range(self.nbSamples):
                self.model.addConstr(
                    self.underVol[p][s]
                    >= self.demands[s, p] - self.quantity[p])
                self.model.addConstr(
                    self.overVol[p][s]
                    >= self.quantity[p] - self.demands[s, p])
            # Overage and underage costs
            self.model.addConstr(
             self.underageCosts[p] == self.underCosts[p]
             * gp.quicksum(self.underVol[p][s] * self.weights[s]
                           for s in range(self.nbSamples)))
            self.model.addConstr(
                self.overageCosts[p] == self.overCosts[p]
                * gp.quicksum(self.overVol[p][s] * self.weights[s]
                              for s in range(self.nbSamples)))

        # ---- Define objective function ----
        self.obj = gp.quicksum(self.underageCosts[p] + self.overageCosts[p]
                               for p in range(self.nbProducts))
        self.model.setObjective(self.obj, GRB.MINIMIZE)

    def solve(self):
        self.model.optimize()
        # Read optimal solution and store in dict
        zOpt = dict()
        for p in range(self.nbProducts):
            zOpt[p] = self.quantity[p].X
            assert zOpt[p] >= 0
        totalOrder = sum(zOpt.values())
        # Assert that the budget constraint is respected
        try:
            assert totalOrder <= (self.budget + 1e-8)
        except AssertionError:
            print('Error in solving Newsvendor model.')
            print('Total order is {} but budget is {}.'.format(
                totalOrder, self.budget))
            print('Order vector:', zOpt)
            raise
        return zOpt

    def costFunction(self, quantity, demand, s=None):
        assert len(quantity) == len(demand)
        assert len(quantity) == self.nbProducts
        overageCosts = sum(
            [self.overCosts[p] * max(quantity[p] - demand[p], 0)
             for p in range(self.nbProducts)])
        underageCosts = sum(
            [self.underCosts[p] * max(demand[p] - quantity[p], 0)
             for p in range(self.nbProducts)])
        return overageCosts + underageCosts

    def riskFunctional(self, quantity, demand, weights):
        """ Expected value """
        indices = range(len(demand))
        expectedCost = np.sum(
            [self.costFunction(quantity, demand[y]) * weights[y]
             for y in indices])
        return expectedCost


# Define functions used in Newsvendor experiments
def truncated_normal(mean=0, sd=1, low=0, upp=np.inf):
    """ Returns a truncated normal distribution. """
    return truncnorm((low - mean) / sd, (upp - mean) / sd,
                     loc=mean, scale=sd)


def sample_newsvendor_features(d_x, nbSamples, nbSpurious=0):
    """ Sample from independent normal distributions. """
    return np.random.normal(size=(nbSamples, d_x+nbSpurious))


def sample_newsvendor_demand(X, d_x, nbProducts):
    """
    Sample demand from truncated distribution with parameters
    depending on the features in X.
    """
    n = len(X)
    Y = np.zeros((n, nbProducts))
    for i in range(n):
        for k in range(nbProducts):
            # Get index of feature corresponding to product
            if k < ((nbProducts-1) / 2):
                cond_std = np.exp(X[i, 0])
            else:
                cond_std = np.exp(X[i, 1])
            # Sample demand from conditonal distribution
            MEAN = 3
            Y[i, k] = truncated_normal(
                low=0, upp=100, mean=MEAN, sd=cond_std).rvs()
    return Y


def sample_newsvendor_data(nbProducts, nbSamples, nbSpurious=0):
    """ Sample features and demand. """
    d_x = 2
    # Sample points in feature space
    X = sample_newsvendor_features(d_x, nbSamples, nbSpurious=nbSpurious)
    # Sample random demand
    Y = sample_newsvendor_demand(X, d_x, nbProducts)
    return X, Y
