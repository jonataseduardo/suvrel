import numpy as np
from itertools import combinations
from scipy.spatial.distance import squareform, pdist
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt


class SUVREL(BaseEstimator):
    """
    Supervised Relevance Learning (SUVREL). 

    Geometric approach for feature selection and dimentionallity reduction. The SUVREL 
    method estimate a metric tensor on the feature space that minimize a cost fuction
    that penalizes large intraclass distances and favors small interclass
    distances.        

    Parameters
    ----------

    n_components: int, float or None (default return all compoents)
        Number of components to keep.

    gamma: float (default 2), this parameter balances the inter and intra classes cost
        functions

    normalization: None, ``variance`` (default), ``t-test``.
        normalize the metric tensor ``variance`` the ``t-test``
        normalization is only possible 



    """

    def __init__(self, n_components=None, gamma=2, normalization=None):
        self.n_components = n_components
        self.gamma = gamma
        self.normalization = normalization

    def fit(self, X, y):
        """

        
        Parameters
        ----------
        X : array like ot sparse matrix of shape [n_samples, n_features]

        y : array-like of shape [n_samples,]
            Target values.

        """

        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        assert self.n_classes_ > 1
        if self.n_components is not None:
            assert self.n_components <= X.shape[1]

        if self.normalization is "t-test" and self.n_classes_ != 2 and self.gamma != 2:
            raise ValueError(
                "``t-test`` normalization can only be used"
                "if y has 2 classes gamma = 2"
                "but there is {0} classes in y and got gamma={1}".format(
                    self.n_classes_, self.gamma
                )
            )

        mean_cl = np.zeros((self.n_classes_, self.n_features_))
        for i, cl in enumerate(self.classes_):
            mean_cl[i] = np.mean(X[y == cl], axis=0)
        smeans = np.zeros(self.n_features_)
        for i, j in combinations(range(self.n_classes_), 2):
            smeans += (mean_cl[i] - mean_cl[j]) ** 2

        if self.gamma != 2 or self.normalization is "t-test":
            var_cl = np.zeros((self.n_classes_, self.n_features_))
            for cl in self.classes_:
                var_cl[cl] = np.var(X[y == cl], axis=0)
            svar = np.sum(var_cl, axis=0)

        if self.gamma != 2:
            metric = (self.gamma - 2.0) * svar + self.gamma / (self.n_classes_ - 1) * smeans
        else:
            metric = smeans

        metric[metric < 0] = 0

        if self.normalization is "variance":
            metric = metric / np.var(X, axis=0)
        if self.normalization is "t-test":
            n1 = np.sum([y == self.classes_[i]])
            n2 = np.sum([y == self.classes_[j]])
            tnorm = ((n1 - 1) * var_cl[i] + (n2 - 1) * var_cl[j]) / (n1 + n2 - 2)
            metric = metric / tnorm

        metric = metric / np.sqrt(np.sum(metric ** 2))

        self.feature_index_ = np.argsort(-metric)
        self.metric_ = metric[self.feature_index_]
        return self

    def transform(self, X, y=None):
        """

        Parameters
        ----------
        X : ndarray, sparse matrix of shape (n_samples, n_features)

        y : None, not used

        """

        check_is_fitted(self, "metric_")
        X = check_array(X)
        X = X[:, self.feature_index_]
        return (X * np.sqrt(self.metric_))[:, slice(0, self.n_components)]

    def fit_transform(self, X, y):
        """

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)

        y : None, not used

        """

        self.fit(X, y)
        return self.transform(X, y)

    def suvrel_metric(self):
        """

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)

        y : None, not used

        """

        check_is_fitted(self, "metric_")

        return self.metric_

    def distance_matrix(self, X, y=None):
        """

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)

        y : None, not used

        """

        check_is_fitted(self, "metric_")
        X_suvrel = self.transform(X)
        return euclidean_distances(X_suvrel, X_suvrel)

    def plot_relevance(self, feature_names=None):
        """

        Parameters
        ----------
        feature_names: list of feature names, default None (uses feature index)


        """

        check_is_fitted(self, "metric_")
        if self.n_components is None:
            n = self.n_features_
        else:
            n = self.n_components

        if feature_names is None:
            feature_names = self.feature_index_
        else:
            feature_names = np.array(feature_names)[self.feature_index_]

        fig, ax = plt.subplots()
        ax.barh(range(n), self.metric_[:n], height=0.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()
        ax.set_xlabel("feature relevance")
        fig.tight_layout()
        return ax

