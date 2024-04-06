import numpy as np
import matplotlib.pyplot as plt
import os

## Mesh class
## read in openfoam mesh and compute geometric properties
## to read in mesh the "boundary", "faces", "neighbour", "owner" and "points" files of the openFoam format are required, which have to be located in directory "constant/polyMesh"
## this class reads also the boundary condition in openfoam format. The file has to be located in a directory names "0" and has to be named "phi"
## fixedValue and fixedGradient boundary types are supported
class Mesh:
    def __init__(self):
        self._points_list = [] # coordinates of points
        self._faces_list = [] # faces defined by index of points
        self._geometricCenterVolume_list = [] # geometric center of a finite volume (2d)
        self._centroidVolume_list = [] # centroid of a finite volume (2d)
        self._area_list = [] # area of volume | in 3D volume
        self._centroidFace_list = [] # centroids of faces
        self._faceNormal_list = [] # normal of faces | normals point away from owner
        self._neighbours = [] # 
        self._faceNorm_list = [] # area of face (in 2D length)
        self._owner_list = [] # owner of faces
        self._neighbour_list = [] # neighbour of faces
        self._geomDiff_list = [] # distance between face centroid and volume centroids [owner, neighbour]
        self._dCF_list = [] # distance between neighbour and owner centroid
        self._e = [] # unit vector of E
        self._E = [] # vector for orthogonal like contribution
        self._T = [] # vector for non-orthogonal like contribution
        self._numElements = -1
        self._bndStart = -1

    # compute geometric center of volumes
    def computeGeometricCenterFaceBased(self):
        geometricCenter = np.zeros((self._numElements,2))
        used_points = [[] for _ in range(self._numElements)]
        
        points = self._points_list
        for i, face in enumerate(self._faces_list[:self._bndStart]):
            owner = self._owner_list[i]
            neighbour = self._neighbour_list[i]
            for point in face:
                if point not in used_points[owner]:
                    geometricCenter[owner,0] += points[0,point]
                    geometricCenter[owner,1] += points[1,point]
                    used_points[owner].append(point)
                if point not in used_points[neighbour]:
                    geometricCenter[neighbour,0] += points[0,point]
                    geometricCenter[neighbour,1] += points[1,point]
                    used_points[neighbour].append(point)

        
        for i, face in enumerate(self._faces_list[self._bndStart:]):
            owner = self._owner_list[i + self._bndStart]
            for point in face:
                if point not in used_points[owner]:
                    geometricCenter[owner,0] += points[0,point]
                    geometricCenter[owner,1] += points[1,point]
                    used_points[owner].append(point)

        for i, p in enumerate(used_points):
            geometricCenter[i] /= len(p)

        self._geometricCenterVolume_list = geometricCenter
         
    # compute centroids     
    def computeCentroidFaceBased(self):
        gc = self._geometricCenterVolume_list
        points = self._points_list
        centroid = np.zeros((self._numElements, 2))
        S = np.zeros((self._numElements,))
        for i, face in enumerate(self._faces_list[:self._bndStart]):
            owner = self._owner_list[i]
            neighbour = self._neighbour_list[i]
            co = gc[owner]
            cn = gc[neighbour]
            point1 = points[:,face[0]]
            point2 = points[:,face[1]]
            centerLocalOwner = (point1 + point2 + co) / 3
            centerLocalNeighbour = (point1 + point2 + cn) / 3
            SlocalOwner = np.linalg.norm(0.5 * np.cross(point1 - co, point2 - co))
            SlocalNeighbour = np.linalg.norm(0.5 * np.cross(point1 - cn, point2 - cn))
            centroid[owner,:] += SlocalOwner * centerLocalOwner
            centroid[neighbour,:] += SlocalNeighbour * centerLocalNeighbour
            S[owner] += SlocalOwner
            S[neighbour] += SlocalNeighbour

        for i, face in enumerate(self._faces_list[self._bndStart:]):
            owner = self._owner_list[i+self._bndStart]
            co = gc[owner]
            point1 = points[:,face[0]]
            point2 = points[:,face[1]]
            centerLocalOwner = (point1 + point2 + co) / 3
            SlocalOwner = np.linalg.norm(0.5 * np.cross(point1 - co, point2 - co))
            centroid[owner,:] += SlocalOwner * centerLocalOwner
            S[owner] += SlocalOwner
            
        for i, s in enumerate(S):
            centroid[i] /= s

        self._centroidVolume_list = centroid
        self._area_list = S
        
    # compute values corresponding to faces (normals, length, etc)
    def computeFaceValues(self):
        for face in self._faces_list:
            point1 = self._points_list[:,face[0]]
            point2 = self._points_list[:,face[1]]  
            cg = 0.5 * (point1 + point2)
            self._centroidFace_list.append(cg)
            nv = [point1[1] - point2[1], point2[0] - point1[0]]
            nvNorm = np.linalg.norm(nv)
            nv = nv / nvNorm
            self._faceNormal_list.append(list(nv))
            self._faceNorm_list.append(nvNorm)

        self._faceNorm_list = np.array(self._faceNorm_list)
        self._faceNormal_list = np.array(self._faceNormal_list)
        self._centroidFace_list = np.array(self._centroidFace_list)
        #print(self._faceNormal_list)

    # compute the distance from face midpoint to volume midpoints | [owner, neighbour]
    def computeGeoDiffFaceBased(self):
        for i, face in enumerate(self._faces_list[:self._bndStart]):
            owner = self._owner_list[i]
            neighbour = self._neighbour_list[i]
            
            geomDiff= [-1, -1]
                
            owner_c = self._centroidVolume_list[owner]
            neighbour_c = self._centroidVolume_list[neighbour]
            self._dCF_list.append(np.linalg.norm(owner_c - neighbour_c))
            self._e.append((neighbour_c - owner_c) / self._dCF_list[-1])
            self._E.append(self._faceNorm_list[i]**2 / np.dot(self._e[-1], self._faceNormal_list[i] * self._faceNorm_list[i]) * self._e[-1])
                           
            self._geomDiff_list.append(geomDiff)
        
        for ii, face in enumerate(self._faces_list[self._bndStart:]):
            i = ii + self._bndStart

            owner = self._owner_list[i]
            
            cf = self._centroidFace_list[i]
            geomDiff= [-1, -1]
            
            owner_c = self._centroidVolume_list[owner]
            diff = np.linalg.norm(owner_c - cf)
            geomDiff[0] = diff

            owner_c = self._centroidVolume_list[owner]
            diff = np.linalg.norm(owner_c - cf)
            self._dCF_list.append(diff)
            self._e.append((cf- owner_c) / self._dCF_list[-1])
            self._E.append(self._faceNorm_list[i]**2 / np.dot(self._e[-1], self._faceNormal_list[i] * self._faceNorm_list[i]) * self._e[-1])

            self._geomDiff_list.append(geomDiff)

            self._T = []
            for n, S, E in zip(self._faceNormal_list, self._faceNorm_list, self._E):
                self._T.append(n*S - E)
        
        self._geomDiff_list = np.array(self._geomDiff_list)
        self._e = np.array(self._e)
        self._dCF_list = np.array(self._dCF_list)
        self._E = np.array(self._E)
        self._T = np.array(self._T)

    # plot mesh
    def plotMesh(self):
        plt.figure()
        for face in self._faces_list:
            point0 = self._points_list[:,face[0]]
            point1 = self._points_list[:,face[1]]
            plt.plot([point0[0], point1[0]], [point0[1], point1[1]], color='k')

        plt.xlabel("x in m")
        plt.ylabel("y in m")
        plt.title(f"Mesh with {len(self._elements_list)} elements")
        plt.show()

    # plot mesh and computed values 
    def plotMeshAll(self):
        plt.figure()
        for face in self._faces_list:
            point0 = self._points_list[:,face[0]]
            point1 = self._points_list[:,face[1]]
            plt.plot([point0[0], point1[0]], [point0[1], point1[1]], color='k')

        self._geometricCenterVolume_list = np.array(self._geometricCenterVolume_list)
        self._centroidVolume_list = np.array(self._centroidVolume_list)
        self._centroidFace_list = np.array(self._centroidFace_list)
        self._faceNormal_list = np.array(self._faceNormal_list)
        plt.scatter(self._geometricCenterVolume_list[:,0], self._geometricCenterVolume_list[:,1], color='r', marker=9)
        plt.scatter(self._centroidVolume_list[:,0], self._centroidVolume_list[:,1], color='b', marker=8)
        plt.scatter(self._centroidFace_list[:,0], self._centroidFace_list[:,1], color='g', marker="P")
        
        plt.quiver(self._centroidFace_list[:,0], self._centroidFace_list[:,1], self._faceNormal_list[:,0], self._faceNormal_list[:,1], angles='xy', scale_units='xy', scale=50)
          
        plt.xlabel("x in m")
        plt.ylabel("y in m")
        plt.title(f"Mesh with {len(self._elements_list)} elements")
        plt.show()

    # read openfoam meshfile
    def readOpenFOAM(self):
        # read neighbour file
        rel_path = "neighbour"
        path = os.path.join("constant", "polyMesh")
        abs_file_path = os.path.join(path, rel_path)
        
        numInteriorF = self._bndStart

        f = open(abs_file_path, "r")
        str = f.readlines()
        f.close()

        istart = 0
        maxi = len(str)
        numI = -1
        while True:
            checkStr = str[istart]
            if len(checkStr) >= 2 and checkStr[:-1].isdigit():
                numI = int(str[istart])
                break

            istart += 1
            if istart == maxi:
                break

        istart += 2    
        
        self._neighbour_list = np.zeros((numInteriorF,), np.int64)
        for i in range(numInteriorF):
            self._neighbour_list[i] = int(str[i+istart][:-1])

        # read owner file
        rel_path = "owner"
        
        abs_file_path = os.path.join(path, rel_path)
        
        f = open(abs_file_path, "r")
        strOwner = f.readlines()
        f.close()

        istartOwner = 0
        maxi = len(strOwner)
        while True:
            checkStr = strOwner[istartOwner]
            if len(checkStr) >= 3 and checkStr[:-2].isdigit():
                break

            istartOwner += 1
            if istartOwner == maxi:
                break


        istartOwner += 2    

        # read points file        
        rel_path = "points"
    
        abs_file_path = os.path.join(path, rel_path)
        
        f = open(abs_file_path, "r")
        str = f.readlines()
        f.close()

        istart = 0
        maxi = len(str)
        numI = -1
        while True:
            checkStr = str[istart]
            if len(checkStr) >= 3 and checkStr[:-2].isdigit():
                numI = int(str[istart])
                break

            istart += 1
            if istart == maxi:
                break

        numI = int(numI/2)
        numPoints = numI
        
        istart += 2
        self._points_list = np.zeros((2, numI))

        z_coord = None
        for i, point in enumerate(str[istart:istart+numI]):
    
            p = point[1:-2].split()
            if z_coord == None:
                z_coord = p[-1]

            if p[-1] == z_coord:
                self._points_list[0,i] = float(p[0])
                self._points_list[1,i] = float(p[1])

        # read faces file
        rel_path = "faces"
    
        abs_file_path = os.path.join(path, rel_path)
        
        f = open(abs_file_path, "r")
        str = f.readlines()
        f.close()
        
        self._faces_list = []
        iface = 0
        for face in str:
            if face[0].isdigit() and face[1] == '(':
                f = face[2:-2].split()
                f = np.array([int(fi) for fi in f])
                
                f2 = list(f[f<numPoints])
                ind = [i for i,fi in enumerate(f) if fi in f2]
                
                if len(f2) == 2:
                    if ind[1] == 3:
                        self._faces_list.append([f2[1], f2[0]])
                    else:
                        self._faces_list.append(f2)

                    self._owner_list.append(int(strOwner[iface+istartOwner][:-1]))

                iface += 1

        self._owner_list = np.array(self._owner_list, dtype=np.int64)
        self._numElements = max(max(self._owner_list), max(self._neighbour_list))+1
                    
    # read openfoam bnd file    
    def readBndFile(self):
        rel_path = "boundary"
        path = "constant/polyMesh"
        abs_file_path = os.path.join(path, rel_path)
        
        f = open(abs_file_path, "r")
        str = f.readlines()
        f.close()

        istart = 0
        maxi = len(str)
        numI = -1
        while True:
            checkStr = str[istart]

            if len(checkStr) >= 2 and checkStr[:-1].isdigit():
                numI = int(str[istart])
                break

            istart += 1
            if istart == maxi:
                break

        istart += 2
        bnd = []
        for i in range(numI):
            ii = istart + i * 7
            p = str[ii:ii+7]
            if p[0].split()[0] != "frontAndBack" and p[0].split()[0] != "frontAndBackPlanes":
                patch = {"name": p[0].split()[0],
                    "startFace": int(p[5].split()[-1][:-1]),
                     "numFaces": int(p[4].split()[-1][:-1])}
                bnd.append(patch)

        
        self._bndStart = min([b["startFace"] for b in bnd])

        rel_path = "phi"
        path = "0"

        abs_file_path = os.path.join(path, rel_path)
        
        f = open(abs_file_path, "r")
        str = f.readlines()
        f.close()

        istart = 0
        maxi = len(str)
        numI = -1
        while True:
            checkStr = str[istart]

            if checkStr[:-1] == "boundaryField":
                break

            istart += 1
            if istart == maxi:
                break

        istart += 2
        
        while True:
            checkStr = str[istart][:-1]

            if len(checkStr.split()) == 1 and checkStr.split()[0] != "{" and checkStr.split()[0] != "}":
                name = checkStr.split()[0]
                bndIndex = [i for i, b in enumerate(bnd) if b["name"] == name][0]
                bnd[bndIndex]["type"] = str[istart+2].split()[1][:-1]
                bnd[bndIndex]["value"] = float(str[istart+3].split()[1][:-1])
                istart += 4

            istart += 1
            if checkStr == "}":
                break

        self._meshBnd = bnd
    
        