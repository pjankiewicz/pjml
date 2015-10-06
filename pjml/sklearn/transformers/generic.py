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


class SparseNormalizer(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        self.colmaxs = np.asarray(x.max(0).todense())[0]
        self.colmins = np.asarray(x.min(0).todense())[0]
        return self

    def transform(self, x):
        x = x.tocsr().copy()
        x.data = (x.data - self.colmins[x.indices]) / \
            (self.colmaxs[x.indices] - self.colmins[x.indices])
        x.data[np.isnan(x.data)] = 0
        x.data = x.data.clip(0,1)
        return x
