import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

from .generic import Densify

class PandasCategoricalExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        if self.columns is None:
            self.columns = x.columns
        return x.ix[:, self.columns].T.to_dict().values()


class SparseMatrixToDataFrame(BaseEstimator, TransformerMixin):

    def __init__(self, features_map):
        self.features_map = features_map

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        columns = [
            c for c, i in sorted(self.features_map.items(),
                                 key=lambda x: x[1])
        ]
        df = pd.DataFrame(x.todense())
        df.columns = columns
        return df


def pandas_cat_vectorizer(columns=None, sparse=False):
    if sparse:
        pipeline = Pipeline([
            ('pd_converter', PandasCategoricalExtractor(columns=columns)),
            ('vect', DictVectorizer()),
        ])
    else:
        pipeline = Pipeline([
            ('pd_converter', PandasCategoricalExtractor(columns=columns)),
            ('vect', DictVectorizer()),
            ('dense',Densify())
        ])
    return pipeline


class PandasFeatureConversion(BaseEstimator, TransformerMixin):

    def __init__(self, func, columns=None, add_new=""):
        self.columns = columns
        self.func = func
        assert add_new is not None and type(add_new) == str, \
            "add_new must be a string (new column name)"
        self.add_new = add_new

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x_ = x.copy()
        if self.columns is None:
            self.columns = x.columns
        for col in self.columns:
            x_[col + self.add_new] = x_[col].map(self.func)
        return x_


class PandasSelector(BaseEstimator, TransformerMixin):

    def __init__(self, dtype=None, columns=None, inverse=False, verbose=False):
        self.dtype = dtype
        self.columns = columns
        self.inverse = inverse
        self.verbose = verbose

    def check_condition(self, x, col):
        cond = (self.dtype is not None and x[col].dtype == self.dtype) or \
               (self.columns is not None and col in self.columns)
        return self.inverse ^ cond

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        selected_cols = []
        for col in x.columns:
            if self.check_condition(x, col):
                selected_cols.append(col)
        if len(selected_cols) == 1:
            return np.squeeze(np.array(x.ix[:, selected_cols].values))
        else:
            return x.ix[:, selected_cols]


class PandasImputeMissing(BaseEstimator, TransformerMixin):

    def __init__(self, default):
        self.default = default

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.fillna(self.default)


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
