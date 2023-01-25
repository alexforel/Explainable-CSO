"""
Model and solve a Shortest Path problem with weighted SAA.
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import networkx as nx


class ShortestPath:
    """
    Model and solve a Shortest Path problem.
    """

    def __init__(self, Y_train, gridLength,
                 weights=None, verbose=False):
        """ Initialize model """
        self.L = Y_train.shape[1]
        self.n = Y_train.shape[0]
        self.Y = Y_train
        # Consider a square grid of width self.dim
        self.dim = gridLength
        # Add horizontal edges
        self.Edge_list = [(i, i+1) for i in range(1, self.dim**2 + 1)
                          if i % self.dim != 0]
        # Add vertical edges
        self.Edge_list += [(i, i + self.dim) for i in range(1, self.dim**2 + 1)
                           if i <= self.dim**2 - self.dim]
        # Assigns each edge to a unique integer from 0 to number-of-edges
        self.Edge_dict = {}
        for index, edge in enumerate(self.Edge_list):
            self.Edge_dict[edge] = index
        assert len(self.Edge_dict) == self.L
        self.Edges = gp.tuplelist(self.Edge_list)
        # Define adaptive weights
        if weights is None:
            self.weights = np.ones(self.n)/self.n
        else:
            self.weights = weights
            assert len(weights) == self.n
        self.verbose = verbose
        self.riskType = 'Expectation'

    def build(self):
        """
        Sets up a shortest path decision problem over a grid network,
        where driver starts in northwest corner and tries to find
        shortest path to southeast corner.
        """
        self.model = gp.Model('shortest_path')
        # Define flow variables and constraints
        self.zVar = self.model.addVars(self.Edges, ub=1, name='flow')
        # Flow balance at all non-start/end nodes
        self.model.addConstrs(
            (gp.quicksum(self.zVar[i, j] for i, j in self.Edges.select(i, '*'))
             - gp.quicksum(self.zVar[k, i]
                           for k, i in self.Edges.select('*', i))
             == 0 for i in range(2, self.dim**2)), name='inner_nodes')
        # Flow balance at start node
        self.model.addConstr(
            (gp.quicksum(self.zVar[i, j]
             for i, j in self.Edges.select(1, '*')) == 1),
            name='start_node')
        # Flow balance at terminal node
        self.model.addConstr(
            (gp.quicksum(self.zVar[i, j]
             for i, j in self.Edges.select('*', self.dim**2)) == 1),
            name='end_node')
        # Objective: expected costs
        self.model.setObjective(
            gp.quicksum(
                self.weights[s] * gp.quicksum(
                    self.Y[s, self.Edge_dict[(i, j)]] * self.zVar[i, j]
                    for i, j in self.Edges) for s in range(self.n)),
            GRB.MINIMIZE)
        if not self.verbose:
            self.model.setParam('OutputFlag', 0)

    def solve(self):
        self.model.optimize()
        zOpt = dict()
        for i, j in self.Edges:
            zOpt[self.Edge_dict[(i, j)]] = self.zVar[i, j].X
            # Post-process numerical instabilities
            if -1e-3 < zOpt[self.Edge_dict[(i, j)]] < 1e-3:
                zOpt[self.Edge_dict[(i, j)]] = 0
            elif (1+1e-3) > zOpt[self.Edge_dict[(i, j)]] > (1-1e-3):
                zOpt[self.Edge_dict[(i, j)]] = 1
            else:
                print('Warning: numerical instability at edge ({},{})'.format(
                    i, j))
                print('Gurobi found value: ', zOpt[self.Edge_dict[(i, j)]])
        # Check that the solution is a valid path from first to last node
        self._assert_solution_is_a_path(zOpt)
        return zOpt

    def _assert_solution_is_a_path(self, zOpt):
        G = nx.DiGraph()
        G.add_edges_from(self.Edge_list)
        # Get solution path as a list
        last_node = (self.dim ** 2)
        nodesSol = []
        for index, edge in enumerate(self.Edge_list):
            if zOpt[self.Edge_dict[edge]] == 1:
                nodesSol.append(edge[0])
                if edge[1] == last_node:
                    nodesSol.append(last_node)
        nodesSol = sorted(nodesSol)  # Sort list
        try:
            assert nodesSol[0] == 1
            assert nodesSol[-1] == last_node
            assert nx.is_simple_path(G, nodesSol)
        except AssertionError:
            print('Error: the solution of CVaR shortest-path')
            print('is not a path from node the first to last node!')
            print('Node path: ', nodesSol)
            raise

    def costFunction(self, z, y, s=None):
        assert len(z) == len(y)
        assert len(z) == 2 * (self.dim-1)*self.dim
        return sum(np.array(list(z.values())) * y)

    def riskFunctional(self, z, y, weights):
        """ Expected value """
        indices = range(len(y))
        expectedCost = np.sum(
            [self.costFunction(z, y[i]) * weights[i]
             for i in indices])
        return expectedCost
