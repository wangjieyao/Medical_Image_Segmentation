import numpy as np

from sklearn.neighbors import KDTree
from sklearn.neighbors.base import KNeighborsMixin, NeighborsBase
from sklearn.metrics import accuracy_score


class FuzzyKNN(NeighborsBase, KNeighborsMixin):
    def __init__(self, k_neighbors=3, leaf_size=30, metric='minkowski', n_jobs=1):
        self.k = k_neighbors
        super().__init__(
            n_neighbors=k_neighbors,
            leaf_size=leaf_size, metric=metric, p=2,
            n_jobs=n_jobs)

    def fit(self, X, y=None):
        self._check_params(X, y)
        self.X = X
        self.y = y

        self.xdim = len(self.X[0])
        self.n_samples = y.shape[0]

        self.classes = np.unique(y, return_inverse=True)[0]

        # init for kneighbors.kneighbors
        self._fit_X = X
        self._tree = KDTree(X)
        self._fit_method = 'kd_tree'

        # calculate memberships
        self.memberships = self._cal_memberships()

    def predict(self, X):
        if self.memberships is None:
            raise Exception('predict() called before fit()')
        else:
            eta = 1e-16
            m = 2
            exp = 2 / (m - 1)
            neigh_dist, neigh_ind = self.kneighbors(X)

            # sum(1/(x-x_k) ** exp)
            den = 1 / ((neigh_dist + eta) ** exp)

            # sum(u_(k,j) * (1/ (x-x_k) ** exp))
            weights = self.memberships[neigh_ind]
            weights = weights.T * den.T
            weights = (np.sum(weights, axis=1)).T
            # Iterations take an especially long time. Using matrix operations can significantly save running time
            # weights = np.array([den[i] @ (self.memberships[neigh_ind])[i] for i in range(X.shape[0])])
            normalizer = den.sum(axis=1)[:, np.newaxis]
            weights /= normalizer
            y_pred = self.classes[np.argmax(weights, axis=1)]

            return y_pred, weights

    def score(self, X, y):

        if self.memberships is None:
            raise Exception('score() called before fit()')
        else:
            y_pred, _ = self.predict(X)
            return accuracy_score(y, y_pred)

    def _cal_memberships(self):

        neigh_dist, neigh_ind = self.kneighbors(self.X)
        weights = np.ones_like(neigh_ind)
        _y = self.y

        all_rows = np.arange(self.n_samples)
        pred_labels = _y[:][neigh_ind]
        proba_k = np.zeros((self.n_samples, len(self.classes)))

        # a simple ':' index doesn't work right
        for i, idx in enumerate(pred_labels.T):  # loop is O(n_neighbors)
            proba_k[all_rows, idx] += weights[:, i]

        # normalize 'votes' into real [0,1] probabilities
        normalizer = proba_k.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba_k /= normalizer  # nj/K

        ad = np.zeros(proba_k.shape)
        ad[all_rows, _y] = 0.51
        memberships = proba_k * 0.49 + ad

        return memberships

    def _check_params(self, X, y):
        if type(self.k) != int:
            raise Exception('"k_neighbors" should have type int')
        elif self.k >= len(y):
            raise Exception('"k_neighbors" should be less than the number of feature sets')
        elif self.k % 2 == 0:
            raise Exception('"k_neighbors" should be odd')
        if X is None or y is None:
            raise Exception('"X" or "y" should not be none')

        if len(X) != len(y):
            raise Exception('The number of feature sets "X" should be equal the number of targets "y" ')
