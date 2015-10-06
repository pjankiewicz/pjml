import numpy as np


def normalize_sparse_matrix(mat):
    colmaxes = np.zeros(mat.shape[1], dtype=np.float64)
    for col in range(len(colmaxes)):
        colmaxes[col] = mat[:, col].max()
    mat = mat.tocsr()
    mat.data /= colmaxes[mat.indices]
    mat = mat.tocsc()

    return mat
