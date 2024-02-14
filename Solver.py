import Diff
import Mesh
import numpy as np
import Time
import VTKWRITER
import LinearSolver
import Interpolate
import time
import Conv

class Solver:
    def __init__(self, mesh):
        self._mesh = mesh
        self._diff = None
        self._gamma = None
        self._bnd = None
        self._startBndId = None
        self._timestep = None
        self._numIter = None
        self._T = None
        self._numElements = None
        self._time = None
        self._vtkWriter = None
        self._uFunc = None

    def setBoundary(self, bnd):
        self._bnd = bnd
        self._startBndId = min([patch["startFace"] for patch in bnd])

    def setSimParameter(self, gamma, timestep, numIter):
        self._gamma = gamma
        self._timestep = timestep
        self._numIter = numIter

    def loadMatrix(self):
        self._diff = Diff.Diff(self._mesh)
        self._diff.setSimParameter(self._gamma)
        self._diff.setBoundary(self._bnd)
        self._diff.buildMatrix()
        self._numElements = self._diff._numElements

        self._time = Time.Time(self._mesh)
        self._time.setSimParameter(self._timestep)


    def initField(self, Tinit):
        self._T = []
        for x in self._mesh._centroidVolume_list:
            self._T.append(Tinit(x[0], x[1]))        
        
    def runSimulation(self):
        self._vtkWriter = VTKWRITER.VTKWRITER(self._mesh, "simulation")
        self._vtkWriter.writeScalar(self._T, 0)
        interp = Interpolate.Grad(self._mesh, self._bnd)
        interp.updateT(self._T)
        interp.interpolate()
        interp.gradientCenter()    
        interp.gradientInterpolate()
        self._vtkWriter.writeVector(interp._gradT, 0)
        
        for i in range(self._numIter):
            self._time.buildMatrix(self._T)
            
            A = self._diff._A + self._time._A
            b = self._diff._b + self._time._b
            
            ls = LinearSolver.LinearSolver(A,b,self._diff, interp,  self._T, 1e-6, 1000, 0.6)
            ls.solve()
            self._T = ls._T
        
            self._vtkWriter.writeScalar(self._T, i+1)
            interp.updateT(self._T)
            interp.interpolate()            
            interp.gradientCenter()    
            interp.gradientInterpolate()
            self._vtkWriter.writeVector(interp._gradT, i+1)

    def simConvection(self):
        self._vtkWriter = VTKWRITER.VTKWRITER(self._mesh, "simulation")
        self._vtkWriter.writeScalar(self._T, 0)
        conv = Conv.Conv(self._mesh)
        conv.setSimParameter(self._uFunc)
        conv.setBoundary(self._bnd)
        conv.buildMatrix()

        self._T = np.linalg.solve(conv._A, conv._b)
        self._vtkWriter.writeScalar(self._T, 0)

    def setVelocity(self, uFunc):
        self._uFunc = uFunc
        
        
    