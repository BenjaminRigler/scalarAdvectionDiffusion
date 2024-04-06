import numpy as np
import copy

# class defining solver for linear systems of equations
# using sparse matrices and the SOR method
class LinearSolver:
    # initialize
    def __init__(self, A, b, time, diff, grad, conv, Tinit, tol, maxIter, lamb_relaxation):
        self._A = A
        self._b = np.array(b, dtype=np.double)
        self._numElements = len(Tinit)
        self._T = np.array(Tinit[:], dtype=np.double)
        self._tol = tol
        self._maxIter = maxIter
        self._lamb_relaxation = lamb_relaxation
        self._diff = diff
        self._grad = grad
        self._conv = conv
        self._time = time
        
    # solve system
    def solve(self):    
        
        numVol = self._numElements
        iter = 0        
        n_1 = numVol-1
        bcor = copy.deepcopy(self._b)
        
        while True:
            iter += 1

            Told = copy.deepcopy(self._T)
            self._grad.updateT(Told)
            self._grad.interpolate()
            self._grad.gradientCenter()
            self._grad.updateFaceValues()
            self._grad.gradientCenter()    
            self._grad.gradientInterpolate()

            bcor[:] = self._b[:]
            if self._diff != None:
                bcor = self._diff.updateBVector(self._grad._gradTint, bcor)
            if self._conv != None:
                bcor = self._conv.updateB(bcor, self._T, self._grad._gradT)
            
            for i in range(numVol-1):
                index = self._A.rowPtr[i:i+2]
                values = self._A.val[index[0]:index[1]]
                colId = self._A.colInd[index[0]:index[1]]
                self._T[i] = self._T[i] + self._lamb_relaxation * ((-np.sum(values[colId!=i] * self._T[colId[colId!=i]]) + bcor[i]) / values[colId==i][0] - self._T[i]) 
            
            index = self._A.rowPtr[numVol-1:numVol+1]
            values = self._A.val[index[0]:]
            colId = self._A.colInd[index[0]:]
            self._T[n_1] = self._T[n_1] + self._lamb_relaxation * ((-np.sum(values[colId!=n_1] * self._T[colId[colId!=n_1]]) + bcor[n_1]) / values[colId==n_1][0] - self._T[n_1])
             
            norm = np.linalg.norm(self._T - Told)
            if norm < self._tol:
                print("Gauss-Seidel solver reached convergence")
                break
            if iter == self._maxIter:
                print("Gauss-Seidel solver diverged")
                break
            if iter%100 == 0:
                print(f"GS: Res = {norm}")
        