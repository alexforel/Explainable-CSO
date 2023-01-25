import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.Prescriptor import Prescriptor


class KnnPrescriptor(Prescriptor):
    """
    Use a k-NN prescriptor to determine adaptive contextual weights
    and output a contextually optimal prescription.

    Implement data-driven k-NN prescriptors as in:
        Bertsimas, D., & Kallus, N. (2020). From predictive to
        prescriptive analytics. Management Science, 66(3), 1025-1044.
    """

    def __init__(self, X_train, Y_train, ProblemModel, gurobiEnv,
                 k=1, useDual=False):
        Prescriptor.__init__(self, ProblemModel, useDual)
        self.X_train = X_train
        self.Y_train = Y_train
        self.gurobiEnv = gurobiEnv
        self.nbSamples = len(X_train)
        self.k = k
        # Fit scaler and train k-NN
        self.x_train_scaled = self._fit_scaler(X_train)
        self._fit_k_nearest_neighbors()

    def _fit_k_nearest_neighbors(self):
        """ Fit k-NN on scaled training data. """
        self.prescriptor = NearestNeighbors(n_neighbors=self.k, metric='l1')
        self.prescriptor.fit(self.x_train_scaled)

    # - Consistency checks -
    def _check_results_consistency(self, x_sol, lambda_sol, verbose,
                                   costFunction, z_alt, z_opt):
        milpNnIndices = np.nonzero(lambda_sol)[0].tolist()
        self._check_milp_has_k_nn(milpNnIndices)
        self._check_milp_and_sklearn_have_same_knn(
            x_sol, milpNnIndices, verbose)

    def _check_milp_has_k_nn(self, milpNnIndices):
        try:
            assert len(milpNnIndices) == self.k
        except AssertionError:
            print("My MILP finds ", len(milpNnIndices), "neighbours"
                  " instead of ", self.k)
            raise

    def _check_milp_and_sklearn_have_same_knn(self, x_sol,
                                              milpNnIndices, verbose):
        # Read k-NN from MILP solution and from scikitlearn
        prescriptorNNIndices = self.prescriptor.kneighbors(
            x_sol, return_distance=False)[0].tolist()
        assert len(prescriptorNNIndices) == self.k
        if verbose:
            print("Foumd the nearest neighbors of alternative context.")
            print("Prescriptor: ", np.sort(prescriptorNNIndices))
            print("My MILP:     ", np.sort(milpNnIndices))
        if not set(milpNnIndices) == set(prescriptorNNIndices):
            print("Warning: MILP and scikitlearn finds different neighbors.")
            print("Prescriptor: ", np.sort(prescriptorNNIndices))
            print("My MILP:     ", np.sort(milpNnIndices))
            # Find indices of different neighbors
            milpUniqueIndices = list(set(milpNnIndices)
                                     - set(prescriptorNNIndices))
            scikitUniqueIndices = list(set(prescriptorNNIndices)
                                       - set(milpNnIndices))
            # Measure distance to these neighbors
            milpAvgDist = np.sum(
                np.abs(x_sol - self.x_train_scaled[milpUniqueIndices]))
            scikitAvgDist = np.sum(
                np.abs(x_sol - self.x_train_scaled[scikitUniqueIndices]))
            if np.abs(milpAvgDist - scikitAvgDist) < 1e-6:
                print("This might be due to numerical instability.")
            # Report distance and raise error
            if scikitAvgDist >= milpAvgDist:
                print("Scikitlearn finds incorrect k-nearest neighbors.")
            else:
                print("! - My MILP finds incorrect k-nearest neighbors - !")
                print("My MILP distance to unique neighbors:    ",
                      milpAvgDist)
                print("scikitlearn distance to unique neighbors:",
                      scikitAvgDist)

    # - Public methods -
    def prescriptor_weights(self, context):
        """
        Find k-nearest neighbors and give them equal weights.
        """
        weights = np.zeros(self.nbSamples)
        kNNIndices = self.prescriptor.kneighbors(
            self.scaler.transform(context), return_distance=False)[0]
        assert len(kNNIndices) == self.k
        for index in kNNIndices:
            weights[index] = 1/self.k
        assert np.abs(1 - sum(weights)) < 1e-8
        return weights
