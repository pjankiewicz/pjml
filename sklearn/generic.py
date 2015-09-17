from collections import Counter, OrderedDict
from itertools import cycle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline


class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class Densify(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.todense()


class CategoryToNumerical(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, default_value=-1000, order_type="random",
                 random_seed=1):
        self.columns = columns
        self.default_value = default_value
        self.order_type = order_type
        np.random.seed(seed=random_seed)

    def zigzag(self, vs):
        vs = list(vs[:])
        spins = cycle([0, -1])
        for spin in spins:
            try:
                v = vs[spin]
                del vs[spin]
                yield v
            except IndexError:
                break

    def order_values(self, values):
        if self.order_type == "random":
            values = np.random.permutation(values)
        elif self.order_type == "sorted_by_value":
            values = sorted(values)
        elif self.order_type == "sorted_by_freq":
            cnt = Counter(values)
            values = zip(*cnt.most_common())[0]
        elif self.order_type == "sorted_by_freq_zigzag":
            cnt = Counter(values)
            values = zip(*cnt.most_common())[0]
            values = list(self.zigzag(values))
        else:
            raise ValueError, "% order type not implemented" % \
                self.order_type
        return values

    def fit(self, x, y=None):
        ts = np.zeros(x.shape)
        self.unique_values = []

        if self.columns is None:
            self.columns = x.columns

        for col in self.columns:
            values = np.unique(x[col])
            values = self.order_values(values)
            d = dict(zip(values, np.arange(len(values))))
            self.unique_values.append(d)
        return self

    def transform(self, X):
        ts = np.zeros((X.shape[0], len(self.columns)), dtype=np.int)
        for i, (d, col) in enumerate(zip(self.unique_values, self.columns)):
            ts[:, i] = [d.get(v, self.default_value) for v in X[col]]
        return ts
