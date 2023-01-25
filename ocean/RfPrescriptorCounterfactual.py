from gurobipy import GRB, quicksum
import numpy as np
# Import OCEAN functions and classes
from ocean.PrescriptorCounterfactual import PrescriptorCounterfactualMilp
from ocean.RandomForestCounterfactual import RandomForestCounterfactualMilp
from ocean.RandomAndIsolationForest import RandomAndIsolationForest


class RfPrescriptorCounterfactualMilp(PrescriptorCounterfactualMilp,
                                      RandomForestCounterfactualMilp):
    def __init__(
            self, prescriptor, sample,
            z_Opt, z_Alt, costDifference,
            isSampleInTreeLeaf, nbSamplesInLeaf, gurobiEnv,
            objectiveNorm=2, isolationForest=None, verbose=False,
            featuresType=False, featuresPossibleValues=False,
            featuresActionnability=False, oneHotEncoding=False,
            absoluteExplanation=False,
            useDual=False):
        self.randomCostsActivated = False
        # Instantiate the Milp: implements actionability constraints
        # and feature consistency according to featuresType
        PrescriptorCounterfactualMilp.__init__(
            self, prescriptor, sample,
            z_Opt, z_Alt, costDifference,
            prescriptor.randomForest, isSampleInTreeLeaf, nbSamplesInLeaf,
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
        RandomForestCounterfactualMilp.__init__(self)
        assert len(self.clf.feature_importances_) == self.nFeatures
        self.model.modelName = "RfPrescriptorCounterFactualMilp"
        # Combine random forest and isolation forest into a completeForest
        self.isolationForest = isolationForest
        self.completeForest = RandomAndIsolationForest(
            self.clf, isolationForest)
        if self.riskType == 'CVaR':
            self.W_MAX = self.prescriptor.W_MAX

    # ---------------------- Private methods ------------------------
    def __get_leaf_indices(self, tm):
        leavesIndices = []
        for v in range(tm.n_nodes):
            if tm.is_leaves[v]:
                leavesIndices.append(v)
        return leavesIndices

    def __addTreeWeightConstraint(self):
        """ Add tree weight constraint:
        for each sample, add constraint to link y_{t,v} to ω_{t,i}
        """
        # Initialize tree weights decision variables
        self.omegaVar = dict()
        for t in self.completeForest.randomForestEstimatorsIndices:
            self.omegaVar[t] = dict()
            for i in range(self.nbSamples):
                self.omegaVar[t][i] = self.model.addVar(
                    lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                    name="omega_t"+str(t)+"i"+str(i))
        # Add constraint to link tree weights to tree leaves
        for t in self.completeForest.randomForestEstimatorsIndices:
            tm = self.treeManagers[t]
            leavesIndices = self.__get_leaf_indices(tm)
            # Add constraint for weight of tree t on sample i
            for i in range(self.nbSamples):
                self.model.addConstr(self.omegaVar[t][i] == sum(
                    [(tm.y_var[v] * self.isSampleInTreeLeaf[t][v][i]
                      / self.nbSamplesInLeaf[t][v])
                     for v in leavesIndices]))

    def __addSampleWeightConstraint(self):
        """ Determine average sample weight:
        average the tree sample weights over the forest.
        """
        treeIndices = self.completeForest.randomForestEstimatorsIndices
        # Average over all trees in the forest
        for i in range(self.nbSamples):
            self.model.addConstr(
                len(treeIndices) * self.wVar[i]
                == quicksum(self.omegaVar[t][i] for t in treeIndices))

    # - Absolute explanation constraints -
    def _add_lazy_leaves_exclusion_constraint(self, leafIndices):
        """ Add constraint to lead to absolute explanation.
        The constraint cuts the set of leaves from the
        previous relative explanation solution. """
        treeIndices = self.completeForest.randomForestEstimatorsIndices
        # Collect indices
        tmList = []
        for t in treeIndices:
            tmList.append(self.treeManagers[t])
        # Add constraint
        self.model.cbLazy(
            quicksum(tmList[t].y_var[leafIndices[t]] for t in treeIndices)
            <= len(leafIndices) - 1)

    # -- Check model status and solution --
    def __checkResultPlausibility(self):
        x_sol = np.array(self.x_sol, dtype=np.float32)
        if self.isolationForest.predict(x_sol)[0] == 1:
            if self.verbose:
                print("Result is an inlier")
        else:
            assert self.isolationForest.predict(x_sol)[0] == -1
            print("Result is an outlier")

    def _get_indices_of_active_leaves(self, y_var):
        # Read set of leaves used in solution
        leafIndicesSol = dict()
        for t in self.completeForest.randomForestEstimatorsIndices:
            tm = self.treeManagers[t]
            activeLeaves = []
            leavesIndices = self.__get_leaf_indices(tm)
            for v in leavesIndices:
                if y_var[t, v] > 1e-4:
                    activeLeaves.append(v)
            if not len(activeLeaves) == 1:
                print("Incorrect set of active leaves:")
                print(activeLeaves)
                raise(ValueError)
            leafIndicesSol[t] = activeLeaves[0]
        return leafIndicesSol

    # -- Callback --
    def absolute_explanation_callback(self, model, where):
        if where == GRB.Callback.MIPSOL:
            z_star, isAbsExplanation = self._check_incumbent_abs_explanation()
            if not isAbsExplanation:
                # - Prescriptor specific -
                # Add constraint to remove the leaf from the feasible domain
                y_var = dict()
                for t in self.completeForest.randomForestEstimatorsIndices:
                    tm = self.treeManagers[t]
                    leavesIndices = self.__get_leaf_indices(tm)
                    for v in leavesIndices:
                        y_var[t, v] = self.model.cbGetSolution(tm.y_var[v])
                leafIndicesSol = self._get_indices_of_active_leaves(y_var)
                self._add_lazy_leaves_exclusion_constraint(leafIndicesSol)
                # - General valid inequality -
                self._add_risk_valid_inequality(z_star)

    # -- Build and read methods --
    def _add_prescriptor_constraints(self):
        # Specific to prescriptor type
        self.buildForest()
        self.__addTreeWeightConstraint()
        self.__addSampleWeightConstraint()

    def _read_prescriptor_solutions(self):
        """ Read indices of active leaves. """
        y_var = dict()
        for t in self.completeForest.randomForestEstimatorsIndices:
            tm = self.treeManagers[t]
            leavesIndices = self.__get_leaf_indices(tm)
            for v in leavesIndices:
                y_var[t, v] = tm.y_var[v].getAttr(GRB.Attr.X)
        self.leafIndicesSol = self._get_indices_of_active_leaves(y_var)
