"""
Auxiliary functions to run numerical experiments.
"""
# Import local functions
from src.RfPrescriptor import RfPrescriptor
from src.KnnPrescriptor import KnnPrescriptor


def print_decisions(x_init, z_alt, z_opt, x_alt=None):
    print('Context x_init :', x_init)
    if x_alt is not None:
        print(' True alternative contex x_alt:', x_alt)
    print(' Solution of CSO problem in context x_init:', z_opt)
    print(' Alternative solution in alternative context:', z_alt)


def print_explanations(x_init, x_rel, x_abs, x_alt=None):
    print('Context x_init:', x_init)
    print(' Relative explanation:', x_rel)
    print(' Absolute explanation:', x_abs)
    if x_alt is not None:
        print(' True alternative contex:', x_alt)


def run_sensitivity_experiment(simulator, x_init, x_alt, ProblemModel,
                               gurobiEnv, prescriptorType=None,
                               nbTrees=None, max_depth=4,
                               nbNeighbors=None, useIsolationForest=False,
                               useDual=False, cvarOrder=2, random_state=None,
                               verbose=False):
    # Train a prescriptor
    if prescriptorType == 'rf':
        prescriptor = RfPrescriptor(simulator.X_train, simulator.Y_train,
                                    ProblemModel, gurobiEnv, nbTrees=nbTrees,
                                    max_depth=max_depth,
                                    cvarOrder=cvarOrder,
                                    useDual=useDual,
                                    random_state=random_state)
        isRandomForestPrescriptor = True
    elif prescriptorType == 'knn':
        prescriptor = KnnPrescriptor(simulator.X_train, simulator.Y_train,
                                     ProblemModel, gurobiEnv,
                                     k=nbNeighbors, useDual=useDual)
        isRandomForestPrescriptor = False
    else:
        print('Error: the prescriptor is not recognized: ', prescriptorType)
        raise
    # Solve CSO problem
    z_opt = prescriptor.solve_cso_problem(x_init, simulator)
    z_alt = prescriptor.solve_cso_problem(x_alt, simulator)
    # Explain decisions
    x_rel, time_relative = prescriptor.solve_explanation_problem(
            x_init, z_opt, z_alt,
            isRandomForestPrescriptor=isRandomForestPrescriptor,
            getAbsoluteExplanation=False,
            useIsolationForest=useIsolationForest,
            verbose=verbose)
    x_abs, time_absolute = prescriptor.solve_explanation_problem(
            x_init, z_opt, z_alt,
            isRandomForestPrescriptor=isRandomForestPrescriptor,
            getAbsoluteExplanation=True,
            useIsolationForest=useIsolationForest,
            verbose=verbose)
    return z_opt, z_alt, x_rel, x_abs, time_relative, time_absolute


def shipment_filter_first_stage(experimentName, z):
    if experimentName == 'shipment':
        return z[0]
    else:
        return z


def assert_all_elements_equal(df, string, nbElements):
    x = df[string].tolist()[-nbElements:]
    assert x.count(x[0]) == len(x)
