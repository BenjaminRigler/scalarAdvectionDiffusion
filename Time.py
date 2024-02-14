import Mesh
import numpy as np

class Time:
    def __init__(self, mesh):
        self._mesh = mesh
        self._timestep = None
        self._A = None
        self._b = None

    def setSimParameter(self, timestep):
        self._timestep = timestep

    def buildMatrix(self, Told):
        numElements = max(max(self._mesh._owner_list), max(self._mesh._neighbour_list))+1 # consider only using max(owner) for openfoam meshes
        A = np.zeros((numElements, numElements))
        b = np.zeros((numElements,))
        volume = self._mesh._area_list
        for i in range(numElements):
            A[i,i] = volume[i] / self._timestep
            b[i] = volume[i] / self._timestep * Told[i]

        self._A = A
        self._b = b
        