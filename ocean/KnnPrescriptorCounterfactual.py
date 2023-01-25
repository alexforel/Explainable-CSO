from gurobipy import GRB, quicksum
import gurobipy as gp
import numpy as np
# Import OCEAN functions and classes
from ocean.PrescriptorCounterfactual import PrescriptorCounterfactualMilp


class KnnPrescriptorCounterfactualMilp(PrescriptorCounterfactualMilp):
    def __init__(
            self, prescriptor, sample, k, x_train,
            z_Opt, z_Alt, costDifference, gurobiEnv,
            objectiveNorm=2, verbose=False,
            featuresType=False, featuresPossibleValues=False,
            featuresActionnability=False, oneHotEncoding=False,
            isolationForest=None,
            absoluteExplanation=False,
            useDual=False):
        self.randomCostsActivated = False
        self.x_train = x_train
        # Instantiate the Milp: implements actionability constraints
        # and feature consistency according to featuresType
        PrescriptorCounterfactualMilp.__init__(
            self, prescriptor, sample, z_Opt, z_Alt,
            costDifference, None, None, None,
            gurobiEnv,
            objectiveNorm=objectiveNorm,
            verbose=verbose,
            featuresType=featuresType,
            featuresPossibleValues=featuresPossibleValues,
            featuresActionnability=featuresActionnability,
            oneHotEncoding=oneHotEncoding,
            absoluteExplanation=absoluteExplanation,
            useDual=useDual)
        # Instantiate the RandomForestCounterfactualMilp object
        self.k = k
        self.model.modelName = "KnnPrescriptorCounterFactualMilp"
        self.W_MAX = 1/k

    # ---------------------- Private methods ------------------------
    def _all_samples_distance(self):
        self.trainDistVar = dict()
        for i in range(len(self.x_train)):
            absDistVar = self._sample_distance(self.x_train[i])
            # Sum distance over all features
            self.trainDistVar[i] = self.model.addVar(
                lb=0.0, ub=self.nFeatures, vtype=GRB.CONTINUOUS)
            self.model.addConstr(
                self.trainDistVar[i]
                == quicksum(absDistVar[f] for f in range(self.nFeatures)))

    def _sample_distance(self, sample):
        absDistVar = dict()
        assert self.nFeatures == len(sample)
        assert self.nFeatures == len(self.x_var_sol)
        for f in range(self.nFeatures):
            DIST_BOUND = 1.0
            # Measure distance between x and historical sample
            distance = self.model.addVar(
                lb=-DIST_BOUND, ub=DIST_BOUND, vtype=GRB.CONTINUOUS)
            self.model.addConstr(distance == self.x_var_sol[f] - sample[f])
            # Measure absolute distance using Gurobipy gp.abs_ function
            absDistVar[f] = self.model.addVar(
                lb=0.0, ub=DIST_BOUND, vtype=GRB.CONTINUOUS)
            self.model.addConstr(absDistVar[f] == gp.abs_(distance))
        return absDistVar

    def _add_nearest_neighbors_constraints(self):
        self.lambdaVar = dict()
        BIG_M = self.nFeatures
        EPSILON = 1e-3
        # Initialize decison variables:
        #       λ[i] = 1, if i ∈ kNN(x)
        for i in range(self.nbSamples):
            self.lambdaVar[i] = self.model.addVar(
                vtype=GRB.BINARY, name="lambda_i"+str(i))
        # Add auxiliary variable
        self.freeDistVar = self.model.addVar(
            vtype=GRB.CONTINUOUS, name="d")
        # Implement constraints to find nearest neighbor
        for i in range(self.nbSamples):
            self.model.addConstr(
                self.trainDistVar[i]
                <= self.freeDistVar + (1 - self.lambdaVar[i]) * BIG_M)
            self.model.addConstr(
                self.trainDistVar[i]
                >= self.freeDistVar + EPSILON - self.lambdaVar[i] * BIG_M)
        # Allow only k neighbors
        self.model.addConstr(
            quicksum(self.lambdaVar[i] for i in range(self.nbSamples))
            == self.k)

    def _add_sample_weight_constraint(self):
        """
        Assign weight 1/k to a sample if it is a k-nearest neighbor.
        """
        for i in range(self.nbSamples):
            self.model.addConstr(self.k * self.wVar[i] == self.lambdaVar[i])

    # -- Callback --
    def absolute_explanation_callback(self, model, where):
        if where == GRB.Callback.MIPSOL:
            z_star, isAbsExplanation = self._check_incumbent_abs_explanation()
            if not isAbsExplanation:
                # - Prescriptor specific -
                # Add constraint to remove neighbors fromn feasible domain
                wSolDict = self.model.cbGetSolution(self.wVar)
                w_sol = np.fromiter(wSolDict.values(), dtype=float)
                knnIndices = np.nonzero(w_sol)[0].tolist()
                self._add_lazy_neighbors_exclusion_constraint(knnIndices)
                # - General valid inequality -
                self._add_risk_valid_inequality(z_star)

    def _add_lazy_neighbors_exclusion_constraint(self, neighborsIndices):
        """
        Add constraint to lead to absolute explanation.
        The constraint cuts the previous set of neighbors
        from the feasible space.
        """
        # Add constraint
        self.model.cbLazy(quicksum(self.lambdaVar[i]
                                   for i in neighborsIndices) <= self.k - 1)

    # -- Build and read methods --
    def _add_prescriptor_constraints(self):
        # Set up objective function
        self.initObjective()
        # Add relative explanation constraints
        self._all_samples_distance()
        self._add_nearest_neighbors_constraints()
        self._add_sample_weight_constraint()

    def _read_prescriptor_solutions(self):
        """ Read the indices of the nearest neighbours. """
        self.lambda_sol = []
        for i in range(self.nbSamples):
            lambdaVal = self.lambdaVar[i].getAttr(GRB.Attr.X)
            if lambdaVal < 1e-3:
                lambdaVal = 0
            elif lambdaVal > (1 - 1e-3):
                lambdaVal = 1
            else:
                print('Numerical error of lambdaVar at index ', i)
                print('lambda = ', lambdaVal)
                raise
            self.lambda_sol.append(lambdaVal)
