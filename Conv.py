import numpy as np
import copy

class Conv:
    def __init__(self, mesh):
        self._u = None
        self._bnd = None
        self._startBndId = -1
        self._mesh = mesh
        self._A = None
        self._numElements = -1

    def setSimParameter(self, u):
        self._u = []
        facesCenter = self._mesh._centroidFace_list
        for faces in facesCenter:
            self._u.append(u(faces[0], faces[1]))

        self._u = np.array(self._u)

    def buildMatrix(self):
        faces = self._mesh._faces_list
        owner = self._mesh._owner_list
        neighbour = self._mesh._neighbour_list
        u = self._u
        normal = self._mesh._faceNormal_list * self._mesh._faceNorm_list[:,np.newaxis]
        l = self._l
        k = self._k

        self._A = np.zeros((self._numElements, self._numElements))
        self._b = np.zeros((self._numElements,))
        for i, face in enumerate(faces[:self._startBndId]):
            uS = np.dot(u[i,:], normal[i,:])
            posMax = max(uS, 0)
            negMax = max(-uS, 0)

            self._A[owner[i], owner[i]] += l * posMax - k * negMax
            self._A[owner[i], neighbour[i]] += k * posMax - l * negMax

            self._A[neighbour[i], neighbour[i]] += l * negMax - k * posMax
            self._A[neighbour[i], owner[i]] += k * negMax - l * posMax

        for patch in self._bnd:
            startId = patch["startFace"]
            endId = startId + patch["numFaces"] - 1
            for i, face in enumerate(faces[startId:endId+1]):
                idFace = startId + i
                idElement = max(owner[idFace], neighbour[idFace])
                n = None
                if owner[idFace] != -1:
                    n = normal[idFace,:]
                elif neighbour[idFace] != -1:
                    n = - normal[idFace,:]
                
                if patch["type"] == "D":
                    self._b[idElement] -= np.dot(u[idFace,:], n) * patch["value"]
                elif patch["type"] == "N":
                    self._A[idElement, idElement] += np.dot(u[idFace,:], n) 

    def updateB(self, b, T, gradT):
        faces = self._mesh._faces_list
        owner = self._mesh._owner_list
        neighbour = self._mesh._neighbour_list
        faceCenters = self._mesh._centroidFace_list
        cv = self._mesh._centroidVolume_list
        u = self._u
        normal = self._mesh._faceNormal_list * self._mesh._faceNorm_list[:,np.newaxis]
        l = self._l
        k = self._k
        #print(gradT)
        #print(b)
        bnew = copy.deepcopy(b)
        
        for i, face in enumerate(faces[:self._startBndId]):
            uS = np.dot(u[i,:], normal[i,:])
            posMax = max(uS, 0)
            negMax = max(-uS, 0)

            TuPlus = T[neighbour[i]] - 2*np.dot(gradT[owner[i]], (cv[neighbour[i]] - cv[owner[i]]))
            TuNeg = T[owner[i]] - 2*np.dot(gradT[neighbour[i]], cv[owner[i]] - cv[neighbour[i]])
            bnew[owner[i]] += (1-l-k) * negMax * TuNeg#posMax * T[owner[i]]
            bnew[owner[i]] -= (1-l-k) * posMax * TuPlus#negMax * T[neighbour[i]]
            #print(i,  cv[owner[i]] - cv[neighbour[i]])
            bnew[neighbour[i]] += (1-l-k) * posMax * TuPlus #negMax * T[neighbour[i]]
            bnew[neighbour[i]] -= (1-l-k) * negMax * TuNeg

            
        #print(bnew)
        return bnew

    def setBoundary(self, bnd):
        self._bnd = bnd
        self._startBndId = min([patch["startFace"] for patch in bnd])
        owner = self._mesh._owner_list
        neighbour = self._mesh._neighbour_list
        self._numElements = max(max(owner), max(neighbour))+1

    def setScheme(self,l,k):
        self._l = l
        self._k = k