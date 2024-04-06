import numpy as np

# define sparse matrix in CSR format
class SparseMatrix:
    def __init__(self, A):
        self.val = A[np.where(A!=0)].flatten()
        self.rowPtr = np.cumsum(np.count_nonzero(A, axis=1))
        self.rowPtr = np.append(self.rowPtr, self.rowPtr[-1])
        self.rowPtr[1:-1] = self.rowPtr[:-2]
        self.rowPtr[0] = 0
        self.colInd = np.nonzero(A)[1]