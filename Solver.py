import Diff
import Mesh
import numpy as np
import Time
import VTKWRITER
import LinearSolver
import Interpolate
import time
import Conv
import Sparse

# class defining different solver
class Solver:
    # init with mesh
    def __init__(self, mesh):
        self._mesh = mesh
        self._diff = None
        self._gamma = None
        self._bnd = mesh._meshBnd
        self._startBndId = mesh._bndStart
        self._timestep = None
        self._numIter = None
        self._T = None
        self._numElements = mesh._numElements
        self._time = None
        self._vtkWriter = None
        self._uFunc = None
        self._name = None
        self._outputInterval = None

    # set simulation related parameter
    def setSimParameter(self, gamma, timestep, numIter):
        self._gamma = gamma
        self._timestep = timestep
        self._numIter = numIter

    # build different matrices
    def loadMatrix(self, l, k):
        self._diff = Diff.Diff(self._mesh)
        self._diff.setSimParameter(self._gamma)
        self._diff.buildMatrix()

        self._time = Time.Time(self._mesh)
        self._time.setSimParameter(self._timestep)
        self._time.buildMatrix()

        if self._uFunc is not None:
            self._conv = Conv.Conv(self._mesh)
            self._conv.setSimParameter(self._uFunc)
            self._conv.setScheme(l,k)
            self._conv.buildMatrix()

    # initialize scalar field
    def initField(self, Tinit):
        self._T = []
        for x in self._mesh._centroidVolume_list:
            self._T.append(Tinit(x[0], x[1]))     

    # set ouput related parameters
    def setOutput(self, name, iter):
        self._name = name
        self._outputInterval = iter   
        
    # set velocity field
    def setVelocity(self, uFunc):
        self._uFunc = uFunc

    # set numerical related parameter
    def setNumPar(self, rel, tol, maxIterSolver):
        self._rel = rel
        self._tol = tol
        self._maxIterSolver = maxIterSolver

    # diffusion solver
    def runDiffusion(self):
        time_list = [0]
        time = 0
        self._vtkWriter = VTKWRITER.VTKWRITER(self._mesh, self._name)
        self._vtkWriter.writeGeometry(0)
        self._vtkWriter.writeDataHeader(0)
        self._vtkWriter.writeScalar(self._T, 0, "Phi")

        interp = Interpolate.Grad(self._mesh, self._bnd)
        interp.updateT(self._T)
        interp.interpolate()
        interp.gradientCenter()    
        interp.gradientInterpolate()
        self._vtkWriter.writeVector(interp._gradT, 0, "gradPhi")
        
        A = Sparse.SparseMatrix(self._diff._A + self._time._A)
        
        for i in range(self._numIter):
            time += self._timestep

            self._time.buildLoadVector(self._T)
            b = self._diff._b + self._time._b
            
            ls = LinearSolver.LinearSolver(A, b, self._time, self._diff, interp,  None, self._T, self._tol, self._maxIterSolver, self._rel)
            ls.solve()
    
            self._T = ls._T

            if (i+1)%self._outputInterval == 0:
                time_list.append(time)
                self._vtkWriter.writeGeometry(i+1)
                self._vtkWriter.writeDataHeader(i+1)
                self._vtkWriter.writeScalar(self._T, i+1, "Phi")
                interp.updateT(self._T)
                interp.interpolate()            
                interp.gradientCenter()    
                interp.gradientInterpolate()
                self._vtkWriter.writeVector(interp._gradT, i+1, "gradPhi")

        self._vtkWriter.writeSeries(time_list)

        
    # convection solver
    def runConvection(self):
        time_list = [0]
        time = 0

        self._vtkWriter = VTKWRITER.VTKWRITER(self._mesh, self._name)
        self._vtkWriter.writeGeometry(0)
        self._vtkWriter.writeDataHeader(0)
        self._vtkWriter.writeScalar(self._T, 0, "Phi")

        interp = Interpolate.Grad(self._mesh, self._bnd)
        A = Sparse.SparseMatrix(self._conv._A + self._time._A)
        
        for i in range(self._numIter):
            time += self._timestep
    
            self._time.buildLoadVector(self._T)
            b = self._conv._b + self._time._b
            
            ls = LinearSolver.LinearSolver(A, b, self._time, None, interp, self._conv, self._T, self._tol, self._maxIterSolver, self._rel)
            ls.solve()
            self._T = ls._T

            if (i+1)%self._outputInterval == 0:
                time_list.append(time)            
                self._vtkWriter.writeGeometry(i+1)
                self._vtkWriter.writeDataHeader(i+1)
                self._vtkWriter.writeScalar(self._T, i+1, "Phi")

        self._vtkWriter.writeSeries(time_list)
            
    # advection diffusion solver
    def runAdvDiff(self):
        
        self._vtkWriter = VTKWRITER.VTKWRITER(self._mesh, self._name)
        
        self._vtkWriter.writeGeometry(0)
        self._vtkWriter.writeDataHeader(0)
        self._vtkWriter.writeScalar(self._T, 0, "Phi")
        

        interp = Interpolate.Grad(self._mesh, self._bnd)
        interp.updateT(self._T)
        interp.interpolate()
        interp.gradientCenter()    
        interp.gradientInterpolate()
        self._vtkWriter.writeVector(interp._gradT, 0, "gradPhi")

        interp = Interpolate.Grad(self._mesh, self._bnd)
        
        A = Sparse.SparseMatrix(self._diff._A + self._time._A + self._conv._A)
        bb = self._diff._b + self._conv._b
        
        time = 0
        time_list = [0]
        for i in range(self._numIter):
            
            time += self._timestep

            self._time.buildLoadVector(self._T)
            b = bb + self._time._b
            
            ls = LinearSolver.LinearSolver(A, b, time, self._diff, interp, self._conv, self._T, self._tol, self._maxIterSolver, self._rel)

            ls.solve()
            self._T = ls._T
        
            if (i+1)%self._outputInterval == 0:
                time_list.append(time)
                self._vtkWriter.writeGeometry(i+1)
                self._vtkWriter.writeDataHeader(i+1)
                self._vtkWriter.writeScalar(self._T, i+1, "Phi")

                interp.updateT(self._T)
                interp.interpolate()            
                interp.gradientCenter()    
                interp.gradientInterpolate()
                self._vtkWriter.writeVector(interp._gradT, i+1, "gradPhi")

        self._vtkWriter.writeSeries(time_list)

        
    