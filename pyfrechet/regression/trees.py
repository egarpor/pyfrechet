from dataclasses import dataclass
from typing import Generator, Optional, Literal, Union

from sklearn.cluster import KMeans

from pyfrechet.metric_spaces import MetricData
from pyfrechet.metric_spaces.utils import *
from .weighting_regressor import WeightingRegressor


@dataclass
class HonestIndices:
    fit_idx: np.ndarray
    predict_idx: np.ndarray


@dataclass
class Split:
    feature_idx: int
    threshold: float
    impurity: float


@dataclass
class Node:
    selector: HonestIndices
    split: Optional[Split]
    left: Optional['Node']
    right: Optional['Node']


def _2means_propose_splits(X_j):
    """
    Propose a split for splitting variable X_j based in 2-means clustering.

    See Capitaine et al. (2020) and Bulté et al. (2023)
    """
    kmeans = KMeans(
        n_clusters=2,
        n_init=1, # Number of times KMeans is run with different centroid seed
        max_iter=10 # Maximum number of iterations of the KMeans in a single run
    ).fit(X_j.reshape((X_j.shape[0], 1)))

    assert not kmeans.labels_ is None, "2means clustering labels are None"
    sel = kmeans.labels_.astype(bool)

    # KMeans.cluster_centers_ is of shape (n_clusters, n_features=1)
    # and contains the coordinates of the cluster centroids
    if kmeans.cluster_centers_[0, 0] < kmeans.cluster_centers_[1, 0]:
        try:
            split_val = (np.max(X_j[sel]) + np.min(X_j[~sel])) / 2
            yield split_val
        except ValueError:
            try:
                yield np.max(X_j[sel])
            except ValueError:
                yield np.min(X_j[~sel])

    else:
        try:
            split_val = (np.min(X_j[sel]) + np.max(X_j[~sel])) / 2
            yield split_val
        except ValueError:
            try:
                yield np.min(X_j[sel])
            except ValueError:
                yield np.max(X_j[~sel])



def _greedy_propose_splits(X_j):
    """
    Propose all sample values of the splitting variable X_j as possible splits.
    
    Take care, very computationally demanding.
    """
    for i in range(X_j.shape[0]):
        yield X_j[i]


        


class Tree(WeightingRegressor):
    def __init__(self, 
                 split_type: Literal['greedy', '2means']='greedy',
                 impurity_method: Literal['cart', 'medoid']='cart',
                 mtry: Union[int, None]=None,
                 min_split_size: int=5,
                 is_honest: bool=False,
                 honesty_fraction: float=0.5):
        """
        mtry=None carries out no random feature selection at each split. Otherwise,
        and integer value (mtry<X.shape[1]) selects randomly mtry features at each split.
        """
        super().__init__(precompute_distances=(impurity_method =='medoid'))
        # TODO: parameter constraints, see https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/ensemble/_forest.py#L199
        self.split_type = split_type
        self.impurity_method = impurity_method
        self.mtry = mtry
        self.min_split_size = min_split_size
        self.is_honest = is_honest
        self.honesty_fraction = honesty_fraction
        self.root_node = None

    def _var(self, y: MetricData, sel: np.ndarray):
        """
        Method to compute variances in each node in the splitting process.

        It adjusts the method employed according to the .impurity_method selected.
        sel stands for the logical mask indicating with elements of y have been selected.
        """
        w = sel/sel.sum()
        if self.impurity_method == 'cart':
            return y.frechet_var(weights=w)
        elif self.impurity_method == 'medoid':
            return y.frechet_medoid_var(weights=w)
        else:
            raise NotImplementedError(f'impurity_method = {self.impurity_method}')
 
    # General syntax: Generator[YieldType, SendType, ReturnType]
    def _propose_splits(self, Xj) -> Generator[float, None, None]:
        if self.split_type == 'greedy':
            return _greedy_propose_splits(Xj)
        elif self.split_type == '2means':
            return _2means_propose_splits(Xj)
        else:
            raise NotImplementedError(f'split_type = {self.split_type}')

    def _find_split(self, X, X_hon, y: MetricData, mtry: Union[int, None]) -> Union[None, Split]:
        """
        Method to find the best split according to the choosen splitting criterion.
        """
        N, d = X.shape
        N_hon, _ = X_hon.shape

        # Random Feature Selection
        # A slice with None as limit has no effect (indexes everything)
        tried_features = np.random.permutation(np.arange(d))[:mtry]

        split_imp = np.inf # Best impurity achieved
        split_j = 0 # Index of the best split variable
        split_val = 0 # Split value of the best split

        for j in tried_features:
            for candidate_split_val in self._propose_splits(X[:, j]):

                # Individuals of the splitting node going to the left child node
                sel = X[:, j] < candidate_split_val
                sel_hon = X_hon[:, j] < candidate_split_val

                # Resulting child nodes (with this split) sample sizes
                n_l = sel.sum()
                n_r = N - n_l

                n_l_hon = sel_hon.sum()
                n_r_hon = N_hon - n_l_hon

                # Check min_split_size stopping criterion
                if min(n_l, n_r, n_l_hon, n_r_hon) > self.min_split_size:
                    var_l = self._var(y, sel)
                    var_r = self._var(y, ~sel)
                    impurity = (n_l * var_l + n_r * var_r) / N

                    # Check if the obtained impurity is better than the best so far
                    if impurity < split_imp:
                        split_imp = impurity
                        split_j = j
                        split_val = candidate_split_val

        return None if split_imp is np.inf else Split(split_j, split_val, split_imp)

    def _split_to_idx(self, X, node: Node) -> tuple[HonestIndices]:
        split = node.split # .split is a Split instance
        sel = node.selector # .selector is a HonestIndices instance (with fit_idx and predict_idx attributes)

        # fit part
        left_idx_fit = sel.fit_idx[X[sel.fit_idx, split.feature_idx] < split.threshold]
        right_idx_fit = sel.fit_idx[X[sel.fit_idx, split.feature_idx] >= split.threshold]

        # predict part
        left_idx_pred = sel.predict_idx[X[sel.predict_idx, split.feature_idx] < split.threshold]
        right_idx_pred = sel.predict_idx[X[sel.predict_idx, split.feature_idx] >= split.threshold]
        
        # merge back into HonestIndices
        return (HonestIndices(left_idx_fit, left_idx_pred), HonestIndices(right_idx_fit, right_idx_pred))

    def _init_idx(self, N) -> HonestIndices:
        """
        Initialiser of HonesIndices for the first step of the tree construction algorithm.
        
        If self.is_honest=False both, the fitting and the predicting training data coincide.
        """
        if self.is_honest:
            s = int(self.honesty_fraction * N)
            perm = np.random.permutation(N)
            return HonestIndices(perm[:s], perm[s:])
        else:
            all_idx = np.arange(N)
            return HonestIndices(all_idx, all_idx)

    # WeightingRegressor has 2 abstact methods that need to be defined: .fit and .weights_for
    def fit(self, X, y: MetricData):
        # First apply the parent class WeightingRegressor .fit() method
        super().fit(X, y)

        N = X.shape[0]
        d = X.shape[1]

        mtry = d if self.mtry is None else self.mtry
        if mtry > d:
            raise Exception(f'Invalid Argument: mtry={self.mtry} but covariate dimension is {d}.')
        
        root = Node(self._init_idx(N), None, None, None) # Node(selector, split, left, right)
        self.root_node = root
        queue = [root]
        while queue:
        # A list evaluates to False when it is empty, otherwise it evaluates to False
            node = queue.pop(0)
            split = self._find_split(X[node.selector.fit_idx, :], # Splitting (fitting) subset
                                     X[node.selector.predict_idx, :], # Honest (predicting) subset
                                     y[node.selector.fit_idx], # Only use labels from the splitting part
                                     mtry)
            if split:
                node.split = split
                left_indices, right_indices = self._split_to_idx(X, node)
                # ._split_to_idx return a 2-tuple of HonestIndices instances

                node.left = Node(left_indices, None, None, None)
                node.right = Node(right_indices, None, None, None)
                queue.append(node.left)
                queue.append(node.right)
                # free up space by removing selectors not needed in the nodes
                node.selector = None

        return self

    def _selector_to_weights(self, selector: np.ndarray) -> np.ndarray:
        """
        Returns the weights induced by a selector mask (np.ndarray of booleans).
        They are set to 1 if a given observation is selected and to 0 otherwise.
        This method can be used (see weights_for) to asign the weights induced 
        by a Tree in its leaf nodes.
        """
        weights = np.zeros(self.y_train_.shape[0])
        weights[selector] = 1.0
        return weights

    def weights_for(self, x):
        # Assert that the root node has been set (it is done in .fit())
        assert self.root_node, "No root_node has been initialized"
        node = self.root_node
        while True and node:
            if not node.split:
                # If there is no split in the node we assign the weights corresponding to the
                # predicting part training observations present in such node
                return self._normalize_weights(self._selector_to_weights(node.selector.predict_idx), 
                                               sum_to_one=True, clip=True)
            
            # If there is a split, retain the node in which observation of interest x falls
            elif x[node.split.feature_idx] < node.split.threshold: 
                node = node.left
            else:
                node = node.right