from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import numpy as np
# Load OCEAN functions
from ocean.RfPrescriptorCounterfactual import RfPrescriptorCounterfactualMilp
from ocean.KnnPrescriptorCounterfactual import KnnPrescriptorCounterfactualMilp


class Prescriptor():
    """
    The Prescriptor class contains methods to
    generate machine-learning based prescriptors,
    determine the adaptive contextual weights,
    and solve explanation problems.
    """

    def __init__(self, ProblemModel, useDual):
        # Read problem and type of risk functional
        self.ProblemModel = ProblemModel
        self.useDual = useDual
        # Read informations from decision problem
        tempModel = self.ProblemModel(None)
        self.riskType = tempModel.riskType
        if self.riskType == 'CVaR':
            self.alpha = tempModel.alpha

    # -- Private methods --
    #   Used only within Prescriptor or inherited classes
    def _get_alt_and_opt_risks(self, weights,
                               z_alt, z_opt,
                               Y_train):
        """
        Return the risk of the alternative and optimized decisions
        for given prescriptor weights. The risks of the two
        decisions are calcualted in the same context
        """
        altRisk = self.riskFunctional(z_alt, Y_train, weights)
        optRisk = self.riskFunctional(z_opt, Y_train, weights)
        return altRisk, optRisk

    def _fit_scaler(self, X_train):
        """ Fit a scaler to scale all features in [0,1]. """
        self.scaler = MinMaxScaler()
        self.scaler.fit(X_train)
        x_train_scaled = self.scaler.transform(X_train)
        return x_train_scaled

    def _distance(self, x1, x2, objectiveNorm=1):
        """ Calculate distance d(x_1, x_2). """
        if objectiveNorm == 1:
            return np.sum(np.abs(x1 - x2))
        else:
            print("Norm is not supported.")
            raise ValueError

    def _use_isolation_forest(self):
        # Train an isolation forest
        ilf = IsolationForest(max_samples=min(128, self.nbSamples),
                              n_estimators=100,
                              contamination=0.1)
        ilf.fit(self.x_train_scaled)
        return ilf

    # -- Consistency checks and solution tests --
    def _check_cvar_bounds_from_milp(self, altCvar, optCvar, milp):
        """ Check that milp calculates CVaR correctly. """
        try:
            assert (abs(milp.altCvar_sol - altCvar)/altCvar) < 1e-2
        except AssertionError:
            print("Error calculating CVaR of ALT decision")
            print("CVaR from milp:", milp.altCvar_sol)
            print("My CVaR:", altCvar)
            raise
        try:
            # Here, we check only that milp.optCvar_sol <= optCvar
            # since we use the more efficient CVaR constraint
            assert (milp.optCvar_sol <= (optCvar + 1e-3))
        except AssertionError:
            print("Error calculating CVaR of OPT decision!")
            print("MILP CVaR should be <= than opt CVaR.")
            print("CVaR from milp:", milp.optCvar_sol)
            print("Opt CVaR:", optCvar)
            raise

    def _check_relative_explanation_criterion(
            self, phaseString, weights, z_alt,
            z_opt, Y_train, nbSamples, milp=None, verbose=False):
        """
        Check that alternative contexts satisfies
        the relative explanation criterion. """
        altCost, optCost = self._get_alt_and_opt_risks(
            weights, z_alt, z_opt, Y_train)
        if phaseString == 'init':
            if altCost <= optCost:
                if verbose:
                    print("The alternative solution is at least as"
                          " good as the data-driven solution.")
                    print("Risk of alternative decision: ", altCost)
                    print("Risk of data-driven decision: ", optCost)
                return True
            else:
                return False
        else:
            if self.riskType == 'CVaR':
                self._check_cvar_bounds_from_milp(altCost, optCost, milp)
            try:
                assert altCost/optCost < (1+1e-3)
                return True
            except AssertionError:
                print("Error: Relative explanation criterion not satisfied!")
                print("OPT decision has lower risk than ALT decision.")
                print("Risk of alternative decision in ALT context:", altCost)
                print("Risk of data-driven decision in ALT context:", optCost)
                raise

    # -- Public methods --
    def cost_difference_vector(self, z_alt, z_opt):
        """
        Calculate the vector of sample costs:
            {δ^i(z_alt, z_opt)}_i for i ∈ {1,…,n} .
        """
        return [(self.costFunction(z_alt, self.Y_train[s, :], s)
                 - self.costFunction(z_opt, self.Y_train[s, :], s))
                for s in range(self.nbSamples)]

    def check_absolute_explanation(
            self, z_alt, z_star,
            evalWeights, phaseString='non_final'):
        """
        Check whether the solutions is an absolute explanation.
        """
        # Check if solution is a hard explanation
        optimalCost = self.riskFunctional(z_star, self.Y_train, evalWeights)
        alternativeCost = self.riskFunctional(z_alt, self.Y_train, evalWeights)
        if phaseString == 'non_final':
            isAbsoluteExplanation = (alternativeCost <= optimalCost + 1e-3)
            return isAbsoluteExplanation
        else:
            try:
                assert alternativeCost <= optimalCost + 1e-4
                return True
            except AssertionError:
                print("Error: Hard explanation criterion is not satisfied!")
                print("Alternative decision does not have optimal risk.")
                print("Risk of alternative decision in ALT context:",
                      alternativeCost)
                print("Risk of optimal decision in ALT context:",
                      optimalCost)
                raise

    def solve_cso_problem(self, x, simulator):
        """
        Solve Contextual Stochastic Optimization (CSO) problem
        in context x.
        """
        # Calculate adaptive weights and load optim model
        adaptiveWeights = self.prescriptor_weights(x)
        self.optimModel = simulator.get_model(adaptiveWeights)
        # Read information from optimal model
        self.costFunction = self.optimModel.costFunction
        self.riskFunctional = self.optimModel.riskFunctional
        # Solve weighted SAA problem
        self.optimModel.build()
        z_star = self.optimModel.solve()
        return z_star

    def solve_saa_problem(self, simulator):
        """
        Solve sample average approximation (SAA) of
        non-contextual stochastic problem:
        all sample weights are equal to 1/n.
        """
        # Calculate adaptive weights and load optim model
        self.optimModel = simulator.get_model(None)
        # Solve weighted SAA problem
        self.optimModel.build()
        z_SAA = self.optimModel.solve()
        return z_SAA

    def solve_explanation_problem(
            self, x0, z_opt, z_alt,
            isRandomForestPrescriptor=True, getAbsoluteExplanation=False,
            ProblemModel=None, useIsolationForest=None, verbose=False,
            featuresType=False, featuresPossibleValues=False,
            featuresActionnability=False, oneHotEncoding=False):
        """ Solve explanation problem to find closest alternative context."""
        # Compute prescriptor weights
        weights = self.prescriptor_weights(x0)
        # Check whether the alternative decision already satisfies
        # the relative or absolute explanation criteria
        if getAbsoluteExplanation:
            isAbsoluteExplanation = self.check_absolute_explanation(
                    z_alt, z_opt, weights)
            if isAbsoluteExplanation:
                return x0, 0.
        else:
            isRelExplanation = self._check_relative_explanation_criterion(
                'init', weights, z_alt, z_opt,
                self.Y_train, self.nbSamples)
            if isRelExplanation:
                return x0, 0.
        # Compute sample cost distance as a vector
        costDifference = self.cost_difference_vector(z_alt, z_opt)
        # Isolation forest
        if useIsolationForest:
            isolationForest = self._use_isolation_forest()
        else:
            isolationForest = None
        # Formulate and solve explanation optimization problem
        if isRandomForestPrescriptor:
            milp = RfPrescriptorCounterfactualMilp(
                self, self.scaler.transform(x0),
                z_opt, z_alt, costDifference,
                self.isSampleInTreeLeaf, self.nbSamplesInLeaf,
                self.gurobiEnv,
                objectiveNorm=1, verbose=verbose,
                isolationForest=isolationForest,
                absoluteExplanation=getAbsoluteExplanation,
                useDual=self.useDual,
                featuresType=featuresType,
                featuresPossibleValues=featuresPossibleValues,
                featuresActionnability=featuresActionnability,
                oneHotEncoding=oneHotEncoding)
        else:
            milp = KnnPrescriptorCounterfactualMilp(
                self, self.scaler.transform(x0), self.k, self.x_train_scaled,
                z_opt, z_alt, costDifference,
                self.gurobiEnv,
                objectiveNorm=1, verbose=verbose,
                isolationForest=isolationForest,
                absoluteExplanation=getAbsoluteExplanation,
                useDual=self.useDual,
                featuresType=featuresType,
                featuresPossibleValues=featuresPossibleValues,
                featuresActionnability=featuresActionnability,
                oneHotEncoding=oneHotEncoding)
        milp.buildModel()
        isSolved = milp.solveModel()
        if not isSolved:
            return 'No explanation found!', milp.runTime
        # Read solution and compute weights
        x_sol_unscaled = self.scaler.inverse_transform(milp.x_sol)
        return x_sol_unscaled, milp.runTime
