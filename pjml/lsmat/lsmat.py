import array
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib

from tqdm import tqdm

from random import shuffle
from sparse import save_sparse_csr, load_sparse_csr

class LSMAT():

    def __init__(self, X, y, ids, columns):
        self.X = X
        self.y = np.array(y)
        self.ids = np.array(ids)
        self.columns = np.array(columns)

    def save(self, path):
        X = self.X.tocsr()
        save_sparse_csr(path, X)
        joblib.dump((self.y, self.ids, self.columns), path + "_info", compress=5)

    def __getitem__(self, arg):
        (rows,cols) = arg
        row_start = rows.start or 0
        row_end = rows.stop or X.shape[0]
        col_start = cols.start or 0
        col_end = cols.stop or X.shape[1]
        return LSMAT(
            self.X[row_start:row_end, col_start:col_end].copy(), 
            self.y[row_start:row_end].copy(), 
            self.ids[row_start:row_end].copy(),
            self.columns[col_start:col_end].copy())

    @staticmethod
    def load(path):
        X = load_sparse_csr(path + ".npz")
        X = X.tocsc()
        (y, ids, columns) = joblib.load(path + "_info")
        return LSMAT(X, y, ids, columns)


class LSMATColumnSparsityFilter(BaseEstimator, TransformerMixin):

    def __init__(self, min_nnz):
        self.min_nnz = min_nnz
        self.nnz = None

    def fit(self, lsmat):
        self.nnz = lsmat.X.getnnz(axis=0)
        return self

    def transform(self, lsmat):
        selected = np.where(self.nnz >= self.min_nnz)
        return lsmat[selected,:]



def save_ffm(lsmat, save_as):
    fields = [c.split(":")[0] for c in lsmat.columns]
    fields_set = list(set([c.split(":")[0] for c in lsmat.columns]))
    field_to_ind = dict((c,i) for i,c in enumerate(fields_set))
    col_to_field_value = ["{}:{}".format(field_to_ind[f],i) for i,f in enumerate(fields)]
    X_ = lsmat.X.tocsr()

    rows_ind = list(range(X_.shape[0]))
    shuffle(rows_ind)

    fp_X = open(save_as + ".ffm", "w")
    fp_y = open(save_as + "_y.txt", "w")
    fp_ids = open(save_as + "_id.txt", "w")
    for indr in tqdm(range(X_.shape[0])):
        rowind = np.arange(X_.indptr[indr],X_.indptr[indr+1])
        rowbuilder = [str(lsmat.y[indr])]
        for col, val in zip(X_.indices[rowind], X_.data[rowind]):
            rowbuilder.append("{}:{}".format(col_to_field_value[col],val))
        fp_X.write(" ".join(rowbuilder) + "\n")
        fp_y.write("{}\n".format(lsmat.y[indr]))
        fp_ids.write("{}\n".format(lsmat.ids[indr]))
    fp_X.close()
    fp_y.close()
    fp_ids.close()

def convert_to_m1p1(y):
    return -1 if y == 0 else 1

def save_svmlight(lsmat, save_as):
    fields = [c.split(":")[0] for c in lsmat.columns]
    fields_set = list(set([c.split(":")[0] for c in lsmat.columns]))
    field_to_ind = dict((c,i) for i,c in enumerate(fields_set))
    col_to_field_value = ["{}:{}".format(field_to_ind[f],i) for i,f in enumerate(fields)]
    X_ = lsmat.X.tocsr()

    rows_ind = list(range(X_.shape[0]))
    shuffle(rows_ind)

    fp_X = open(save_as + ".svmlight", "w")
    fp_y = open(save_as + "_y.txt", "w")
    fp_ids = open(save_as + "_id.txt", "w")
    for indr in tqdm(range(X_.shape[0])):
        rowind = np.arange(X_.indptr[indr],X_.indptr[indr+1])
        rowbuilder = [str(convert_to_m1p1(lsmat.y[indr]))]
        for col, val in zip(X_.indices[rowind], X_.data[rowind]):
            rowbuilder.append("{}:{}".format(col_to_field_value[col].replace(":",""),val))
        fp_X.write(" ".join(rowbuilder) + "\n")
        fp_y.write("{}\n".format(lsmat.y[indr]))
        fp_ids.write("{}\n".format(lsmat.ids[indr]))
    fp_X.close()
    fp_y.close()
    fp_ids.close()

