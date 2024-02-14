import numpy as np

class Grad:
    def __init__(self, mesh, bnd):
        self._bnd = bnd
        self._mesh = mesh
        self._T = None
        self._startBndId = -1
        self._numElements = max(max(mesh._owner_list), max(mesh._neighbour_list))+1 
        self._Tint = None
        self._gradT = None
        self._gradTint = None

    def interpolate(self):
        self._startBndId = min([patch["startFace"] for patch in self._bnd])
        T = self._T
        owner = self._mesh._owner_list
        neighbour = self._mesh._neighbour_list
        Tint = []
        for i, faces in enumerate(self._mesh._faces_list[:self._startBndId]):
            cf = self._mesh._centroidFace_list[i]
            co = self._mesh._centroidVolume_list[self._mesh._owner_list[i]]
            cn = self._mesh._centroidVolume_list[self._mesh._neighbour_list[i]]
            e = self._mesh._faceNormal_list[i] / np.linalg.norm(self._mesh._faceNormal_list[i])   
            dCf = np.dot(cf-co, e)
            gf = dCf / (dCf + np.dot(cn-cf,e))
            Tint.append(gf * T[self._mesh._neighbour_list[i]] + (1-gf) * T[self._mesh._owner_list[i]])
        
        for patch in self._bnd:
            startId = patch["startFace"]
            endId = startId + patch["numFaces"] - 1
            if patch["type"] == "D":
                for i, face in enumerate(self._mesh._faces_list[startId:endId+1]):
                    Tint.append(patch["value"])
            if patch["type"] == "N":
                for i, face in enumerate(self._mesh._faces_list[startId:endId+1]):
                    idFace = startId + i
                    if owner[idFace] != -1:
                        Tint.append(T[owner[idFace]])
                    if neighbour[idFace] != -1:
                        Tint.append(T[neighbour[idFace]])
        
        self._Tint = Tint

    def gradientCenter(self):
        numElements = self._numElements
        owner = self._mesh._owner_list
        neighbour = self._mesh._neighbour_list
        gradT = np.zeros((numElements, 2))
        Tint = self._Tint
        normal = self._mesh._faceNormal_list * self._mesh._faceNorm_list[:,np.newaxis]
        for i, face in enumerate(self._mesh._faces_list):
            if owner[i] != -1:
                gradT[owner[i]] += Tint[i] *normal[i]
            if neighbour[i] != -1:
                gradT[neighbour[i]] -= Tint[i] *normal[i]
        
        '''
        for patch in self._bnd:
            startId = patch["startFace"]
            endId = startId + patch["numFaces"] - 1
            if patch["type"] == "D":
                for i, face in enumerate(self._mesh._faces_list[startId:endId+1]):
                    idFace = startId + i
                    if owner[idFace] != -1:
                        gradT[owner[idFace]] += patch["value"] *normal[idFace]
                    elif neighbour[idFace] != -1:
                        gradT[neighbour[idFace]] -= patch["value"] *normal[idFace]
        '''
        for i in range(numElements):
            gradT[i] /= self._mesh._area_list[i]
        
        self._gradT = gradT

    def updateT(self, T):
        self._T = T

    def gradientInterpolate(self):
        self._startBndId = min([patch["startFace"] for patch in self._bnd])
        gradT = self._gradT
        gradTint = []
        owner = self._mesh._owner_list
        neighbour = self._mesh._neighbour_list
        for i, faces in enumerate(self._mesh._faces_list[:self._startBndId]):
            cf = self._mesh._centroidFace_list[i]
            co = self._mesh._centroidVolume_list[owner[i]]
            cn = self._mesh._centroidVolume_list[neighbour[i]]
            e = self._mesh._faceNormal_list[i] / np.linalg.norm(self._mesh._faceNormal_list[i])   
            dCf = np.dot(cf-co, e) 
            gf = dCf / (dCf + np.dot(cn-cf,e))
            gradTint.append(gf * gradT[neighbour[i]] + (1-gf) * gradT[owner[i]])

        for patch in self._bnd:
            startId = patch["startFace"]
            endId = startId + patch["numFaces"] - 1
            if patch["type"] == "D":
                for i, face in enumerate(self._mesh._faces_list[startId:endId+1]):                
                    idFace = startId + i
                    if owner[idFace] != -1:
                        gradTint.append(gradT[owner[idFace]])
                    if neighbour[idFace] != -1:
                        gradTint.append(gradT[neighbour[idFace]])

            if patch["type"] == "N":
                for i, face in enumerate(self._mesh._faces_list[startId:endId+1]):
                    idFace = startId + i
                    gradTint.append(patch["value"] * self._mesh._faceNormal_list[i])

        self._gradTint = gradTint

    def updateFaceValues(self):
        owner = self._mesh._owner_list
        neighbour = self._mesh._neighbour_list
        for i, faces in enumerate(self._mesh._faces_list[:self._startBndId]):
            cf = self._mesh._centroidFace_list[i]
            co = self._mesh._centroidVolume_list[self._mesh._owner_list[i]]
            cn = self._mesh._centroidVolume_list[self._mesh._neighbour_list[i]]
            e = self._mesh._faceNormal_list[i] / np.linalg.norm(self._mesh._faceNormal_list[i])   
            dCf = np.dot(cf-co, e)
            gf = dCf / (dCf + np.dot(cn-cf,e))
            rfC = cf - co
            rfF = cf - co
            self._Tint[i] = self._Tint[i] + (1-gf) * np.dot(self._gradT[owner[i]], rfC) + gf * np.dot(self._gradT[neighbour[i]], rfF)
            