import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from src.Prescriptor import Prescriptor


class RfPrescriptor(Prescriptor):
    """
    Trains a random forest prescriptor to determine adaptive contextual weights
    and output a contextually optimal prescription.

    Implement data-driven random forest prescriptors as in:
        Bertsimas, D., & Kallus, N. (2020). From predictive to
        prescriptive analytics. Management Science, 66(3), 1025-1044.
    """

    def __init__(self, X_train, Y_train, ProblemModel,
                 gurobiEnv, nbTrees=100, max_depth=4, cvarOrder=2,
                 random_state=None, isScaled=False, useDual=False):
        Prescriptor.__init__(self, ProblemModel, useDual)
        self.nbTrees = nbTrees
        self.nbSamples = len(X_train)
        self.Y_train = Y_train
        self.gurobiEnv = gurobiEnv
        self.random_state = random_state
        if not isScaled:
            # Fit scaler
            self.x_train_scaled = self._fit_scaler(X_train)
        else:
            self.x_train_scaled = X_train
            self.scaler = DummyScaler()
        # Train a random forest prescriptor
        self._train_random_forest_prescriptor(
            X_train, Y_train, nbTrees,
            max_depth=max_depth, random_state=self.random_state)
        # Read structure of RF prescriptor
        self.trainingLeaves = self.randomForest.apply(self.x_train_scaled)
        self.nbSamplesInLeaf = self._count_nb_samples_in_leaves()
        self.isSampleInTreeLeaf = self._get_matrix_of_sample_leaf_tree()
        if self.riskType == 'CVaR':
            self.cvarOrder = cvarOrder
            self.W_MAX = self._get_bound_on_max_sample_weight()

    # - Private methods -
    def _train_random_forest_prescriptor(self, X_train, Y_train,
                                         nbTrees, max_depth,
                                         random_state=None):
        self.randomForest = RandomForestRegressor(
            n_estimators=nbTrees, max_depth=max_depth, max_features="sqrt",
            random_state=random_state)
        self.randomForest.fit(self.x_train_scaled, Y_train)

    def _count_nb_samples_in_leaves(self):
        """
        Pre-processing: find the matrix S such that
            S[t, v] = the number of training samples in leaf v of tree t
        """
        nbSamplesInLeaf = dict()
        for t in range(self.nbTrees):
            nbSamplesInLeaf[t] = dict()
            leafNodes = []
            # Get tree structure from sklearn RandomForestRegressor
            tree = self.randomForest.estimators_[t].tree_
            nbNodes = tree.node_count
            for v in range(nbNodes):
                isLeaf = (tree.children_left[v] == -1)
                if isLeaf:
                    assert tree.n_node_samples[v] > 0
                    leafNodes.append(v)
                    # Count number of training samples in this leaf
                    nbSamplesInLeaf[t][v] = 0
                    for i in range(self.nbSamples):
                        if (self.trainingLeaves[i, t] == v):
                            nbSamplesInLeaf[t][v] = nbSamplesInLeaf[t][v]+1
        return nbSamplesInLeaf

    def _get_matrix_of_sample_leaf_tree(self):
        """
        Pre-processing: matrix that track the leaves of the training samples
            I[t, v, i] = 1 if sample i is in leaf node v of tree t
        """
        isSampleInTreeLeaf = dict()
        for t in range(self.nbTrees):
            isSampleInTreeLeaf[t] = dict()
            # Get tree structure from sklearn RandomForestRegressor
            tree = self.randomForest.estimators_[t].tree_
            nbNodes = tree.node_count
            for v in range(nbNodes):
                isLeaf = (tree.children_left[v] == -1)
                if isLeaf:
                    isSampleInTreeLeaf[t][v] = np.zeros(self.nbSamples)
                    for i in range(self.nbSamples):
                        if (self.trainingLeaves[i, t] == v):
                            isSampleInTreeLeaf[t][v][i] = 1
                    # Check results
                    try:
                        assert (np.sum(isSampleInTreeLeaf[t][v])
                                == self.nbSamplesInLeaf[t][v])
                    except AssertionError:
                        print("Error identifying samples in leaf", v,
                              " of tree", t)
                        print("Found ", np.sum(isSampleInTreeLeaf[t][v]),
                              ", instead of ",
                              self.nbSamplesInLeaf[t][v])
                        raise
        return isSampleInTreeLeaf

    def _tree_weight(self, treeIndex, leaf, i):
        I_ = self.isSampleInTreeLeaf[treeIndex][leaf][i]
        S_ = self.nbSamplesInLeaf[treeIndex][leaf]
        return (I_ / S_)

    def _create_intersection_graph(self):
        """Calculate matrix [tree, leaves]^2 of leaves intersecting"""
        G = nx.DiGraph()
        # -- 1/ Add arcs from source node to first tree leaves--
        tree = self.randomForest.estimators_[0].tree_
        # Find number of leaves
        nbNodes = tree.node_count
        for v in range(nbNodes):
            isLeaf = (tree.children_left[v] == -1)
            if isLeaf:
                G.add_edge('source', 't_'+str(0)+'_v_'+str(v),
                           weight=1/self.nbSamplesInLeaf[0][v])
        # -- 2/ Add arces between successive trees leaves --
        for t in range(self.nbTrees-1):
            tree = self.randomForest.estimators_[t].tree_
            nextTree = self.randomForest.estimators_[t+1].tree_
            # Find number of leaves
            nbNodes = tree.node_count
            nbNodesNextTree = nextTree.node_count
            for v1 in range(nbNodes):
                isLeaf1 = (tree.children_left[v1] == -1)
                if isLeaf1:
                    samples1 = np.nonzero(self.isSampleInTreeLeaf[t][v1])[0]
                    for v2 in range(nbNodesNextTree):
                        isLeaf2 = (nextTree.children_left[v2] == -1)
                        if isLeaf2:
                            samples2 = np.nonzero(
                                self.isSampleInTreeLeaf[t+1][v2])[0]
                            # Check if leaves "intersect" i.e. have at
                            # least one common sample
                            nodesIntersect = (not set(samples1).isdisjoint(
                                                samples2))
                            if nodesIntersect:
                                G.add_edge(
                                    't_'+str(t)+'_v_'+str(v1),
                                    't_'+str(t+1)+'_v_'+str(v2),
                                    weight=1/self.nbSamplesInLeaf[t+1][v2])
        # -- 3/ Add arcs from last tree leaves to sink node --
        lastIndex = self.nbTrees-1
        tree = self.randomForest.estimators_[lastIndex].tree_
        # Find number of leaves
        nbNodes = tree.node_count
        for v in range(nbNodes):
            isLeaf = (tree.children_left[v] == -1)
            if isLeaf:
                G.add_edge('t_'+str(lastIndex)+'_v_'+str(v),
                           'sink', weight=0)
        return G

    def _get_order_1_w_max(self):
        """  """
        maxTreeWeightsList = [
            1/min(self.nbSamplesInLeaf[t].values())
            for t in range(self.nbTrees)]
        return np.mean(maxTreeWeightsList)

    def _get_bound_on_max_sample_weight(self):
        """
        Determine an upper bound on the maximum weight that
        a sample can have as the average of the tree weights.
        We implement three methods of increasing complexity and
        quality. [order = 0, 1, or 2]
        """
        if self.cvarOrder == 0:
            W_MAX = 1
        elif self.cvarOrder == 1:
            W_MAX = self._get_order_1_w_max()
            assert W_MAX <= 1
        elif self.cvarOrder == 2:
            G = self._create_intersection_graph()
            lengthLongestPath = nx.dag_longest_path_length(
                    G, weight='weight', default_weight=0)
            assert lengthLongestPath > 1
            W_MAX = (1/self.nbTrees) * lengthLongestPath
            # - Check that W_MAX is smaller than 1
            try:
                assert W_MAX <= 1
            except AssertionError:
                print('Error when calculating W_MAX.')
                print('W_MAX = ', W_MAX, ' is > 1 !')
                raise
            # - Check that order 2 method is always better than order 1
            W_o1 = self._get_order_1_w_max()
            try:
                assert W_MAX <= W_o1
            except AssertionError:
                print('Error when calculating W_MAX.')
                print('Order 2 method is worse than order 1 method!')
                print('W_MAX, Order 1: ', W_o1)
                print('W_MAX, Order 1: ', W_MAX)
                raise
        return W_MAX

    # - Public methods -
    def prescriptor_weights(self, context):
        """
        Calculate adaptive contextual weights of
        all historical samples for the current context.
        """
        nbSamples = len(self.trainingLeaves)
        # Type checking and resizing
        if isinstance(context, np.ndarray):
            if len(context) == 1:
                context = context.reshape(1, -1)
        elif isinstance(context, list):
            if len(context) == 1:
                context = np.array(context).reshape(1, -1)
        # Scale context to [0, 1] using MinMaxScaler
        context = self.scaler.transform(context)
        # Get the index of the leaves that contain the context
        leaves = self.randomForest.apply(context)[0]
        # Calculate adaptive weights
        weights = np.zeros(nbSamples)
        for i in range(nbSamples):
            treeWeight = np.zeros(self.nbTrees)
            for t in range(self.nbTrees):
                treeWeight[t] = self._tree_weight(t, leaves[t], i)
            assert np.amax(treeWeight) <= 1
            weights[i] = np.sum(treeWeight) / self.nbTrees
        # Check weights sum to 1
        try:
            assert (np.abs(np.sum(weights) - 1) < 1e-12)
        except AssertionError:
            print("Error calculating weights: weights do not sum to 1.")
            print("Sum of weights = ", np.sum(weights))
            raise
        return weights


class DummyScaler:
    """
    Dummy object to bypass the scaling methods when the
    training data has already been scaled to [0,1].
    """

    def __init__(self):
        pass

    def transform(self, x):
        return np.array(x)

    def inverse_transform(self, x):
        return np.array(x)
