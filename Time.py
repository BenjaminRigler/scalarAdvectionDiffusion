import Mesh
import numpy as np

# Class for time derivative
class Time:
    # initialize class with mesh
    def __init__(self, mesh):
        self._mesh = mesh
        self._numElements = mesh._numElements
        self._timestep = None
        self._A = None
        self._b = None

    # set timestep
    def setSimParameter(self, timestep):
        self._timestep = timestep

    # build A matrix containing only time derivative
    def buildMatrix(self):
        numElements = self._numElements
        A = np.zeros((numElements, numElements))
        volume = self._mesh._area_list
        
        for i in range(numElements):
            A[i,i] = volume[i] / self._timestep

        self._A = A

    # build b vector dependend on scalar from previous timestep
    def buildLoadVector(self, Told):
        numElements = self._numElements
        b = np.zeros((numElements,))
        volume = self._mesh._area_list
        for i in range(numElements):
            b[i] = volume[i] / self._timestep * Told[i]
        self._b = b