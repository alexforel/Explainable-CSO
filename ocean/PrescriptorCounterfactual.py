from gurobipy import GRB, quicksum
from threading import Lock
import numpy as np
from copy import deepcopy
# Import OCEAN utility functions and types
from ocean.CounterfactualMilp import CounterfactualMilp


class PrescriptorCounterfactualMilp(CounterfactualMilp):
    def __init__(self, prescriptor, sample,
                 z_Opt, z_Alt, costDifference,
                 randomForest, isSampleInTreeLeaf, nbSamplesInLeaf,
                 gurobiEnv,
                 objectiveNorm=2,
                 verbose=False,
                 featuresType=False,
                 featuresPossibleValues=False,
                 featuresActionnability=False,
                 oneHotEncoding=False,
                 absoluteExplanation=False,
                 useDual=False):
        CounterfactualMilp.__init__(self, sample, gurobiEnv,
                                    objectiveNorm, verbose, featuresType,
                                    featuresPossibleValues,
                                    featuresActionnability, oneHotEncoding)
        self.prescriptor = prescriptor
        self.riskType = prescriptor.riskType
        if self.riskType == 'CVaR':
            self.alpha = prescriptor.alpha
        # Objects used only by random forest prescriptor
        self.clf = randomForest
        self.isSampleInTreeLeaf = isSampleInTreeLeaf
        self.nbSamplesInLeaf = nbSamplesInLeaf
        # Store parameters of data-driven prescriptor
        self.z_Opt = z_Opt
        self.z_Alt = z_Alt
        self.costDifference = costDifference
        self.nbSamples = len(costDifference)
        self.absoluteExplanation = absoluteExplanation
        self.useDualFormulation = useDual
        if self.absoluteExplanation & (not self.useDualFormulation):
            self.model.Params.lazyConstraints = 1
        self.useRelaxedCvar = True

    # --- Private methods ---
    #   Used only within Prescriptor or inherited classes
    def _init_weight_variables(self):
        """ Initialize sample weights decison variables. """
        self.wVar = dict()
        for i in range(self.nbSamples):
            self.wVar[i] = self.model.addVar(
                lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w_i"+str(i))

    # -- Relative explanation constraints --
    # - Expectation -
    def _add_relative_explanation_constraint(self, costDifference):
        """
        Add constraint that expected cost conditioned on alternative context
        is better for alternative decision than for original optimal decision.
        """
        self.model.addConstr(
            quicksum(self.wVar[i] * costDifference[i]
                     for i in range(self.nbSamples)) <= 0)

    # - CVaR -
    def _formulate_cvar(self, costVector):
        sortedIndices = np.argsort(-costVector)
        # Define variables
        flowVar = dict()
        tauVar = dict()
        for i in range(self.nbSamples):
            tauVar[i] = self.model.addVar(
                lb=0.0, ub=1.0, vtype=GRB.BINARY)
            flowVar[i] = self.model.addVar(
                lb=0.0, ub=1-self.alpha, vtype=GRB.CONTINUOUS)
            # Add domain constraints
            self.model.addConstr(flowVar[i] <= self.wVar[sortedIndices[i]])
        # Determine CVaR
        CVaR = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS)
        self.model.addConstr(
            CVaR == (1/(1-self.alpha)) * quicksum(
                flowVar[i] * costVector[sortedIndices[i]]
                for i in range(self.nbSamples)))
        # Add total flow equal to 1-α
        self.model.addConstr(
            quicksum(flowVar[i] for i in range(self.nbSamples))
            == 1-self.alpha)
        # Add ordering constraints on τ
        for i in range(self.nbSamples-1):
            self.model.addConstr(tauVar[i] >= tauVar[i+1])
        # Link flow and τ variables
        for i in range(self.nbSamples):
            self.model.addConstr(
                self.wVar[sortedIndices[i]]
                - self.W_MAX * (1-tauVar[i]) <= flowVar[i])
            if i > 0:
                self.model.addConstr(
                    flowVar[i] <= tauVar[i-1] * self.W_MAX)
        # Add valid inequality: ordering constraints on cumulative flows
        for j in range(1, self.nbSamples):
            self.model.addConstr(
                quicksum(flowVar[i] for i in range(j))
                <= quicksum(flowVar[i] for i in range(j+1)))
        return CVaR

    def _formulate_relaxed_cvar(self, costVector):
        # Define variables
        flowVar = dict()
        for i in range(self.nbSamples):
            flowVar[i] = self.model.addVar(
                lb=0.0, ub=1-self.alpha, vtype=GRB.CONTINUOUS)
            # Add domain constraints
            self.model.addConstr(flowVar[i] <= self.wVar[i])
        # Determine CVaR
        relaxCVaR = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS)
        self.model.addConstr(
            relaxCVaR == (1/(1-self.alpha)) * quicksum(
                flowVar[i] * costVector[i] for i in range(self.nbSamples)))
        # Add total flow equal to 1-α
        self.model.addConstr(
            quicksum(flowVar[i] for i in range(self.nbSamples))
            == 1-self.alpha)
        return relaxCVaR

    def _add_relative_cvar_constraint(self):
        # Find CVaR(z_alt)
        altCostVector = np.asarray(
            [self.prescriptor.costFunction(
                self.z_Alt, self.prescriptor.Y_train[i, :])
             for i in range(self.nbSamples)], dtype=np.float32)
        self.altCvar = self._formulate_cvar(altCostVector)
        # Find CVaR(z_opt)
        optCostVector = np.asarray(
            [self.prescriptor.costFunction(
                self.z_Opt, self.prescriptor.Y_train[i, :])
             for i in range(self.nbSamples)], dtype=np.float32)
        if self.useRelaxedCvar:
            self.optCvar = self._formulate_relaxed_cvar(optCostVector)
        else:
            self.optCvar = self._formulate_cvar(optCostVector)
        # Add constraint: UB(z_alt) <= LB(z_opt)
        self.model.addConstr(self.altCvar <= self.optCvar)

    # -- Absolute explanation constraints --
    def _check_incumbent_abs_explanation(self):
        """
        Check if incumbent solution is an absolute explanation.
        - Read the incumbent solution
        - Solve CSO problem in this alternative context
        - Check if optimal cost is equal to the one of alternative decision
        """
        evalWeights = np.zeros(self.nbSamples)
        for i in range(self.nbSamples):
            evalWeights[i] = self.model.cbGetSolution(self.wVar[i])
        try:
            assert abs(1 - np.sum(evalWeights)) <= 1e-4
        except AssertionError:
            print('The sample weights do not sum to 1.')
            print('Sum of sample weights: ', np.sum(evalWeights))
            raise
        optModel = self.prescriptor.ProblemModel(evalWeights)
        optModel.build()
        z_star = optModel.solve()
        # Check if solution is a absolute explanation
        isAbsSolution = self.prescriptor.check_absolute_explanation(
                self.z_Alt, z_star, evalWeights)
        return z_star, isAbsSolution

    def _init_cvar_cut_variables(self, NB_CUTS):
        # Define variables
        self.flowVar = dict()
        self.CVaR_var = dict()
        for c in range(NB_CUTS):
            self.CVaR_var[c] = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS)
            self.flowVar[c] = dict()
            for i in range(self.nbSamples):
                self.flowVar[c][i] = self.model.addVar(
                    lb=0.0, ub=1-self.alpha, vtype=GRB.CONTINUOUS)

    def _get_lazy_relaxed_cvar(self, costVector, cutIndex):
        # Constraint on each flow variable: f_i ≤ w_i
        for i in range(self.nbSamples):
            self.model.cbLazy(
                self.flowVar[cutIndex][i] <= self.wVar[i])
        # Add total flow equal to 1-α
        self.model.cbLazy(
            quicksum(self.flowVar[cutIndex][i]
                     for i in range(self.nbSamples)) == 1-self.alpha)
        # Determine CVaR
        self.model.cbLazy(
            self.CVaR_var[cutIndex] == (1/(1-self.alpha)) * quicksum(
                self.flowVar[cutIndex][i] * costVector[i]
                for i in range(self.nbSamples)))
        return self.CVaR_var[cutIndex]

    def _add_risk_valid_inequality(self, z_star):
        """
        Add valid inequality:
            an absolute explanation is always a relative explanation.
        """
        if self.riskType == 'Expectation':
            # Add valid inequality on cost
            costDifference = self.prescriptor.cost_difference_vector(
                    self.z_Alt, z_star)
            self.lock.acquire()
            if z_star not in self.cutList:
                if self.verbose:
                    print('Adding cut for z_star = ', z_star)
                self.cutList.append(z_star)
                self.lock.release()
                self.model.cbLazy(
                    quicksum(self.wVar[i] * costDifference[i]
                             for i in range(self.nbSamples)) <= 0)
            else:
                self.lock.release()
        elif self.riskType == 'CVaR':
            self.lock.acquire()
            # Read cut index and store copy for use in thread
            localCutIndex = deepcopy(self.cutCounter)
            # Increase cut counter and release lock
            self.cutCounter = self.cutCounter + 1
            # Check if intermediate decision is already used in cut
            if ((localCutIndex < self.NB_MAX_CUTS)
                    & (z_star not in self.cutList)):
                self.cutList.append(z_star)
                self.lock.release()
                # Add cut using local cut index
                if self.verbose:
                    print('Adding cut {} out of {} for:'.format(
                        localCutIndex+1, self.NB_MAX_CUTS))
                    print('     z_star = ', z_star)
                zstarCostVector = np.asarray(
                    [self.prescriptor.costFunction(
                        z_star, self.prescriptor.Y_train[i, :])
                     for i in range(self.nbSamples)], dtype=np.float32)
                zstarCVaR = self._get_lazy_relaxed_cvar(
                    zstarCostVector, localCutIndex)
                self.model.cbLazy(self.altCvar <= zstarCVaR)
            else:
                self.lock.release()
        else:
            print('Unknown risk type: ', self.riskType)
            raise ValueError

    def _add_absolute_dual_constraints(self):
        """
        Alternative formulation for absolute explanations
        when the decision problem is linear.
        """
        nbPrimalVars = len(self.z_Opt)
        # Get parameters of decision model: (A^T ⋅ z <= RHS)
        optModel = self.prescriptor.ProblemModel(None)
        optModel.build()
        A = optModel.model.getA()
        nbDualVars = A.shape[0]
        rhs = optModel.model.getAttr("RHS", optModel.model.getConstrs())
        # Define dual variables:
        #       One dual variable per constraint of the primal problem.
        self.dual_var = dict()
        for j in range(nbDualVars):
            self.dual_var[j] = self.model.addVar(lb=-GRB.INFINITY,
                                                 vtype=GRB.CONTINUOUS)
        # Get sumWD = (∑_{i=1}^n w_i d_i) as linear expresssion
        self.sumWD = dict()
        for k in range(nbPrimalVars):
            self.sumWD[k] = quicksum(
                self.wVar[i] * self.prescriptor.Y_train[i, k]
                for i in range(self.nbSamples))
        # Add dual constraints for absolute explanations
        for k in range(nbPrimalVars):
            self.model.addConstr(
                quicksum(A[j, k] * self.dual_var[j]
                         for j in range(nbDualVars)) <= self.sumWD[k])
        self.model.addConstr(
            quicksum(self.sumWD[k] * self.z_Alt[k]
                     for k in range(nbPrimalVars))
            <= quicksum(rhs[j] * self.dual_var[j] for j in range(nbDualVars)))

    # --- Public methods ---
    def buildModel(self):
        # General functions
        self.initSolution()
        self.addActionnabilityConstraints()
        self.addOneHotEncodingConstraints()
        self._init_weight_variables()
        # Specific to prescriptor type
        self._add_prescriptor_constraints()
        if self.absoluteExplanation & self.useDualFormulation:
            self._add_absolute_dual_constraints()
        # Add relative explanation constraint
        if self.riskType == 'Expectation':
            self._add_relative_explanation_constraint(self.costDifference)
        elif self.riskType == 'CVaR':
            self._add_relative_cvar_constraint()
        else:
            print('Unknown risk functional:')
            print('use \'CVaR\' or \'Expectation\'.')
            assert False

    def solveModel(self):
        """ Solve explanation problem. """
        if self.absoluteExplanation & (not self.useDualFormulation):
            self.cutList = []
            self.lock = Lock()
            if self.riskType == 'CVaR':
                self.NB_MAX_CUTS = 5
                self.cutCounter = 0
                self._init_cvar_cut_variables(self.NB_MAX_CUTS)
            self.model.optimize(
                lambda model, where:
                    self.absolute_explanation_callback(model, where))
        else:
            self.model.optimize()
        # Read the optimization results
        isOptimal = self.read_results()
        if not isOptimal:
            return False
        # Read results specific to prescriptor type
        if self.riskType == 'CVaR':
            self.altCvar_sol = self.altCvar.X
            self.optCvar_sol = self.optCvar.X
        self._read_prescriptor_solutions()
        return True
