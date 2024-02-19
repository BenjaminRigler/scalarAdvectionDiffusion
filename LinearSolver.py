import numpy as np
import copy
import Interpolate
import time

class LinearSolver:
    def __init__(self, A, b, diff, grad, conv, Tinit, tol, maxIter, lamb_relaxation):
        self._A = np.array(A, dtype=np.double)
        self._b = np.array(b, dtype=np.double)
        self._numElements = self._A.shape[0]
        self._T = np.array(Tinit[:], dtype=np.double)
        #self._Told = self._T[:]
        self._tol = tol
        self._maxIter = maxIter
        self._lamb_relaxation = lamb_relaxation
        self._diff = diff
        self._grad = grad
        self._conv = conv
        

    def solve(self):
        iter = 0
        while True:
            iter += 1

            Told = copy.deepcopy(self._T)
            self._grad.updateT(Told)
            self._grad.interpolate()
            self._grad.gradientCenter()
            self._grad.updateFaceValues()
            self._grad.gradientCenter()    
            self._grad.gradientInterpolate()
            b = self._diff.updateBVector(self._grad._gradTint, self._b)
            
            for i in range(self._numElements):
                sum = np.sum([self._A[i,j] * self._T[j] for j in range(self._numElements) if (self._A[i,j] != 0 and i != j)])
                self._T[i] =  self._T[i] + self._lamb_relaxation*((-sum + b[i]) / self._A[i,i] - self._T[i])
            n =np.linalg.norm(self._T - Told) 
            if n < self._tol:
                print("Tolerance reached")
                break
            if iter == self._maxIter:
                print("max Iterations reached")
                break
            if iter%100 == 0:
                print(f"Res: {n}")

    def solveConv(self):
        
        iter = 0
        while True:
            iter += 1

            Told = copy.deepcopy(self._T)
            self._grad.updateT(Told)
            self._grad.interpolate()
            self._grad.gradientCenter()
            b = self._conv.updateB(self._b, self._T, self._grad._gradT)
            #print(b)
            #b = copy.deepcopy(self._b)
            for i in range(self._numElements):
                sum = np.sum([self._A[i,j] * self._T[j] for j in range(self._numElements) if (self._A[i,j] != 0 and i != j)])
                self._T[i] =  self._T[i] + self._lamb_relaxation*((-sum + b[i]) / self._A[i,i] - self._T[i])
            
            n = np.linalg.norm(self._T - Told)
            if  n < self._tol:
                print("Tolerance reached")
                break
            if iter == self._maxIter:
                print("max Iterations reached")
                break
            if iter%100 == 0:
                print(f"Res: {n}")
    