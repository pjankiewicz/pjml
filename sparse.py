
import numpy as np


class SparseNormalizer():
    def fit(self,X):
        self.colmaxs = np.asarray(X.max(0).todense())[0]
        self.colmins = np.asarray(X.min(0).todense())[0]

    def transform(self,X):
        X = X.copy()
        X.data = (X.data - self.colmins[X.indices]) / (self.colmaxs[X.indices] - self.colmins[X.indices])            
        return X


def normalize_sparse_matrix(mat):
    colmaxes = np.zeros(mat.shape[1], dtype=np.float64)
    for col in range(len(colmaxes)):
        colmaxes[col] = mat[:, col].max()
    mat = mat.tocsr()
    mat.data /= colmaxes[mat.indices]
    mat = mat.tocsc()

    return mat
