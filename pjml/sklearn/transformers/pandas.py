import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline


class PandasCategoricalExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        if self.columns is None:
            self.columns = x.columns
        return x.ix[:, self.columns].T.to_dict().values()


def pandas_cat_vectorizer(columns=None):
    return Pipeline([
        ('pd_converter', PandasCategoricalExtractor(columns=columns))
        ('vect', DictVectorizer())
    ])


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
        return x.ix[:, selected_cols]


class PandasImputeMissing(BaseEstimator, TransformerMixin):

    def __init__(self, default):
        self.default = default

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.fillna(self.default)
