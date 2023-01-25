import numpy as np
import pandas as pd
import os as os
from sklearn.model_selection import train_test_split
# Load OCEAN functions
from ocean.DatasetReader import DatasetReader
# Load custom functions
from src.Newsvendor import sample_newsvendor_data
from src.Newsvendor import Newsvendor
from src.Shipment import Shipment
from src.ShortestPath import ShortestPath
from src.CvarShortestPath import CvarShortestPath


class Simulator():
    """
    Create a Simulator class to store sample and store training/test data.
    """

    def __init__(self, experimentName, gurobiEnv,
                 verbose=True, nbSamples=100, nbSpurious=0,
                 problemSens=None, d_x=None):
        self.nbSamples = nbSamples
        self.verbose = verbose
        # Sample training set, init counterfactuals and test set
        if experimentName == 'newsvendor':
            """
            Newsvendor with budget
                adapted from Kallus & Mao (2022).
            """
            if problemSens is not None:
                NB_PRODUCTS = problemSens
            else:
                NB_PRODUCTS = 20
            # Sample training data: context and demand
            self.X_train, self.Y_train = sample_newsvendor_data(
                NB_PRODUCTS, self.nbSamples, nbSpurious=nbSpurious)
            # Define sampling function
            self.sample = lambda n=1: sample_newsvendor_data(
                NB_PRODUCTS, n, nbSpurious=nbSpurious)[0]
            # Define function that returns a w-SAA optimization model
            self.get_model = lambda w: Newsvendor(
                self.Y_train, NB_PRODUCTS, gurobiEnv,
                weights=w, verbose=verbose)
        elif experimentName == 'shipment':
            """
            Shipment planning problem:
                taken from Bertsimas & Kallus (2020).
            """
            self.X_train, self.Y_train = self._sample_shipment_data(
                self.nbSamples)
            # Define sampling function
            self.sample = lambda: self._shipment_covariate(
                self._sample_shipment_innovations(),
                um1=self.u[self.nbSamples-2, :],
                um2=self.u[self.nbSamples-3, :],
                xm1=self.X_train[self.nbSamples-2, :],
                xm2=self.X_train[self.nbSamples-3, :])
            # Define function that returns a w-SAA optimization model
            self.get_model = lambda w: Shipment(self.Y_train, self.d_y,
                                                gurobiEnv, weights=w,
                                                verbose=verbose)
        elif experimentName in ['path', 'cvar-path']:
            """
            Noisy shortest path on a square grid
                adapted from Elmachtoub et al. (2020)
            """
            if problemSens is not None:
                GRID_LENGTH = problemSens
            else:
                if experimentName == 'path':
                    GRID_LENGTH = 8
                elif experimentName == 'cvar-path':
                    GRID_LENGTH = 4
            self.nbArcs = 2 * (GRID_LENGTH-1)*GRID_LENGTH
            if d_x is None:
                self.d_x = 4
            else:
                self.d_x = d_x
            self.baseCost = np.ones(self.nbArcs)
            # Sample matrix B \in {0, 1}^{d_y x d_x}
            # The B Matrix specifies which feature impact which arc cost
            self.bMatrix = np.random.binomial(n=1, p=0.5,
                                              size=(self.nbArcs, self.d_x))
            # Sample training data: covariates and arc costs
            self.X_train, self.Y_train = self._sample_path_data(self.nbSamples,
                                                                self.nbArcs)
            # Define sampling function
            self.sample = lambda n=1: self.sample_x_path_problem(n, self.d_x)
            # Define function that returns a w-SAA optimization model
            if experimentName == 'path':
                self.get_model = lambda w: ShortestPath(
                    self.Y_train, GRID_LENGTH, weights=w)
            elif experimentName == 'cvar-path':
                self.get_model = lambda w: CvarShortestPath(
                    self.Y_train, GRID_LENGTH, weights=w)
        elif experimentName == 'uber':
            """
            Shortest-path problem with Uber movement data
                adapted from Kallus & Mao (2022)
            """
            from src.UberPath import UberPath
            # ! SPECIFY YOUR PATH HERE !
            dirPath = os.path.join("C:\\Users\\Documents",
                                   "Code", "explain-cso", "data")
            # Load road network graph
            self.A_mat = pd.read_csv(os.path.join(
                dirPath, "A_downtwon_1221to1256.csv")).to_numpy()
            self.b_vec = pd.read_csv(os.path.join(
                dirPath, "b_downtwon_1221to1256.csv")).to_numpy()
            # Load features and responses
            self.all_X, self.all_Y = self._read_uber_data(dirPath)
            self.nbSamples = len(self.all_X)-1
            self.sample = lambda: self._resample(
                self.all_X, self.all_Y)
            self.get_model = lambda w: UberPath(
                self.Y_train, self.A_mat, self.b_vec,
                weights=w, verbose=self.verbose)
        else:
            print('Experiment name is not available.')
            print('Use: \'newsvendor\', \'shipment\', \'path\', or \'uber\'.')
            raise

    # - Private methods -
    def _sample_shipment_innovations(self):
        return np.random.multivariate_normal(
            np.zeros((self.d_x)), self.uCovar, 1)

    def _shipment_covariate(self, u,
                            um1=np.zeros(3), um2=np.zeros(3),
                            xm1=np.zeros(3), xm2=np.zeros(3)):
        x = (u + self.Theta1.dot(um1) + self.Theta2.dot(um2)
             + self.Phi1.dot(xm1) + self.Phi2.dot(xm2))
        return x

    def _sample_shipment_demand(self, x):
        demand = np.zeros(self.d_y)
        for i in range(self.d_y):
            demand[i] = (self.A[i, :].dot(x + (np.random.normal() / 4))
                         + self.B[i, :].dot(x) * np.random.normal())
        demand = np.maximum(np.zeros(self.d_y), demand)
        return demand

    def _sample_shipment_data(self, nbSamples):
        self.d_x = 3  # number of covariates
        self.d_y = 12  # number of locations
        # Define parameters of ARMA(2,2) process
        self.uCovar = np.array([[1, 0.5, 0.],
                                [0.5, 1.2, 0.5],
                                [0, 0.5, 0.8]])
        self.A = 2.5 * np.array([[0.8, 0.1, 0.1],
                                 [0.1, 0.8, 0.1],
                                 [0.1, 0.1, 0.8],
                                 [0.8, 0.1, 0.1],
                                 [0.1, 0.8, 0.1],
                                 [0.1, 0.1, 0.8],
                                 [0.8, 0.1, 0.1],
                                 [0.1, 0.8, 0.1],
                                 [0.1, 0.1, 0.8],
                                 [0.8, 0.1, 0.1],
                                 [0.1, 0.8, 0.1],
                                 [0.1, 0.1, 0.8]])
        self.B = 7.5 * np.array([[0., -1, -1],
                                 [-1, 0, -1],
                                 [-1, -1, 0],
                                 [0, -1, 1],
                                 [-1, 0, 1],
                                 [-1, 1, 0],
                                 [0, 1, -1],
                                 [1, 0, -1],
                                 [1, -1, 0],
                                 [0, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 0]])
        self.Theta1 = np.array([[0.4, 0.8, 0.],
                                [-1.1, -0.3, 0.],
                                [0, 0, 0]])
        self.Theta2 = np.array([[0, 0.8, 0.],
                                [-1.1, 0, 0.],
                                [0, 0, 0]])
        self.Phi1 = np.array([[0.5, -0.9, 0],
                              [1.1, -0.7, 0],
                              [0, 0, 0.5]])
        self.Phi2 = np.array([[0, -0.5, 0],
                              [-0.5, 0, 0],
                              [0, 0, 0]])
        # Sample covariates X and demand Y
        #       3-dimensional ARMA(2,2) process
        u = np.zeros((nbSamples, self.d_x))
        x = np.zeros((nbSamples, self.d_x))
        y = np.zeros((nbSamples, self.d_y))
        for i in range(nbSamples):
            # Sample innovation
            u[i, :] = self._sample_shipment_innovations()
            # Determine covariate
            if i == 0:
                x[i, :] = self._shipment_covariate(u[i, :])
            elif i == 1:
                x[i, :] = self._shipment_covariate(
                    u[i, :], um1=u[i-1, :], xm1=x[i-1, :])
            else:
                x[i, :] = self._shipment_covariate(
                    u[i, :],
                    um1=u[i-1, :], um2=u[i-2, :],
                    xm1=x[i-1, :], xm2=x[i-2, :])
            # Sample demand
            y[i, :] = self._sample_shipment_demand(x[i, :])
        self.u = u
        return x, y

    def _read_uber_data(self, dirPath):
        """ Load historical features and travel times. """
        self.reader = DatasetReader(
            os.path.join(dirPath, "uber_processed.csv"))
        Y_raw = pd.read_csv(os.path.join(dirPath, "Y_twoyear.csv"))
        all_Y = Y_raw.to_numpy()
        return self.reader.X, all_Y

    def _resample(self, all_X, all_Y):
        self.X_train, x, self.Y_train, _ = train_test_split(
            all_X, all_Y, test_size=1)
        return x

    def _sample_path_data(self, n, nbArcs):
        """
        Generate synthetic data,
        adapted from Section 5.1 of Elmachtoub et al. (2020).
        """
        x_train = self.sample_x_path_problem(n, self.d_x)
        y_train = np.zeros((n, nbArcs))
        noise = np.random.uniform(0, 1, size=(n, nbArcs))
        for i in range(n):
            Bxi = np.matmul(self.bMatrix, x_train[i, :]) / self.d_x
            assert Bxi.shape == (nbArcs,)
            for k in range(nbArcs):
                y_train[i, k] = self.baseCost[k]*Bxi[k]+noise[i, k]
        assert (y_train >= 0).all()
        return x_train, y_train

    def set_start_end_nodes(self, startNode, endNode,
                            census, nodeList):
        """
        Set the start and finish of the shortest path
        over Los Angeles downtown area.
        """
        # Initialize b_vec
        for i in range(len(self.b_vec)):
            self.b_vec[i] = 0
            # Add start and end nodes
            if census[nodeList[i]] == startNode:
                self.b_vec[i] = -1
            elif census[nodeList[i]] == endNode:
                self.b_vec[i] = 1

    def sample_x_path_problem(self, n, d_x):
        return np.random.uniform(0.5, 1.5, size=(n, d_x))
