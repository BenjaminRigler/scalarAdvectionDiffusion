import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy


class Diff:
    def __init__(self, mesh):
        self._mesh = mesh
        self._gamma = None
        self._bnd = None
        self._startBndId = -1
        self._A = None
        self._b = None
        self._T = None
        self._numElements = -1

    def setSimParameter(self, gamma):
        self._gamma = gamma

    def setBoundary(self, bnd):
        self._bnd = bnd
        self._startBndId = min([patch["startFace"] for patch in bnd])
        
    def buildMatrix(self):
        faceArea = self._mesh._faceNorm_list
        faces = self._mesh._faces_list
        owner = self._mesh._owner_list
        neighbour = self._mesh._neighbour_list
        dCF = self._mesh._dCF_list
        E = self._mesh._E
        numElements = max(max(owner), max(neighbour))+1 # consider only using max(owner) for openfoam meshes
        self._numElements = numElements
        geomDiff = self._mesh._geomDiff_list
        self._A = np.zeros((numElements, numElements))
        self._b = np.zeros((numElements,))
       
        for i, face in enumerate(faces[:self._startBndId]):
            flux = self._gamma / dCF[i] * np.linalg.norm(E[i])
            self._A[owner[i], owner[i]] += flux
            self._A[neighbour[i], neighbour[i]] += flux
            self._A[owner[i], neighbour[i]] -= flux
            self._A[neighbour[i], owner[i]] -= flux
            
        for patch in self._bnd:
            startId = patch["startFace"]
            endId = startId + patch["numFaces"] - 1
            for i, face in enumerate(faces[startId:endId+1]):
                idFace = startId + i
                idElement = max(owner[idFace], neighbour[idFace]) # could be changed to only use owner if only openfoam meshes are used
                
                if patch["type"] == "D":
                    #d = max(geomDiff[idFace]) # could be changed to only use owner if only openfoam meshes are used
                    d = dCF[idFace]
                    self._A[idElement, idElement] += self._gamma / d * np.linalg.norm(E[idFace])
                    self._b[idElement] += self._gamma*patch["value"] / d * np.linalg.norm(E[idFace])
                elif patch["type"] == "N":
                    self._b[idElement] += self._gamma * faceArea[i] * patch["value"]
        
    def updateBVector(self, gradT, b):
        faces = self._mesh._faces_list
        T = self._mesh._T
        owner = self._mesh._owner_list
        neighbour = self._mesh._neighbour_list
        bnew = copy.deepcopy(b)
        for i, face in enumerate(faces[:self._startBndId]):
            bnew[owner[i]] += np.dot(gradT[i], T[i]) 
            bnew[neighbour[i]] -= np.dot(gradT[i], T[i])
        return bnew
    
    def solve(self):
        self._T = np.linalg.solve(self._A, self._b)

    def plotSolution(self):
        points = self._mesh._points_list
        data_raw = np.zeros((self._numElements, 4))
        deg = np.zeros((self._numElements))
        owner = self._mesh._owner_list
        neighbour = self._mesh._neighbour_list

        for i, face in enumerate(self._mesh._faces_list):
            point1 = points[0, face[0]]
            point2 = points[1, face[1]]
            if owner[i] != -1:
                data_raw[owner[i], 0] += point1
                data_raw[owner[i], 1] += point2
                deg[owner[i]] += 1
            if neighbour[i] != -1:
                data_raw[neighbour[i], 0] += point1
                data_raw[neighbour[i], 1] += point2
                deg[neighbour[i]] += 1

        for i,d in enumerate(deg):
            data_raw[i,:2] = data_raw[i,:2] / d
            data_raw[i,2] = 0
            data_raw[i,3] = self._T[i]
        
        plt.figure()
        plt.tricontourf(data_raw[:,0], data_raw[:,1], data_raw[:,3])
        plt.show()
        #data = pd.DataFrame(data_raw)
        #data.columns = ['x coord', 'y coord', 'z coord', 'scalar']
        #data.to_csv('results.csv', index=False)

   