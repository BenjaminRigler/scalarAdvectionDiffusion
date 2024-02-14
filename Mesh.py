import numpy as np
import matplotlib.pyplot as plt

class Mesh:
    def __init__(self):
        self._points_list = []
        self._faces_list = []
        self._elements_list = [] #
        self._geometricCenterVolume_list = []
        self._centroidVolume_list = []
        self._area_list = []
        self._centroidFace_list = []
        self._faceNormal_list = []
        self._neighbours = []
        self._faceNorm_list = []
        self._owner_list = []
        self._neighbour_list = []
        self._geomDiff_list = []
        self._dCF_list = []
        self._e = []
        self._E = []
        self._T = []
        self._n = -1
        
    def genSquareMeshHexa(self, L, n):
        self._n = n
        x = np.linspace(0,L, n+1)
        X,Y = np.meshgrid(x,x)
        X = X.flatten()
        Y = Y.flatten()
        self._points_list = np.array([X,Y])


        for j in range(n):
            self._faces_list.append([j+1, j])
        for i in range(n):
            self._faces_list.append([(i+1)*(n+1), i*(n+1)])
            for j in range(n):
                self._faces_list.append([j+1 + (i+1)*(n+1), j+1 + i*(n+1)])
                self._faces_list.append([j+1 + (i+1)*(n+1), j + (i+1)*(n+1)])
        self._faces_list = np.array(self._faces_list)
        
        self._elements_list.append([0, n+1, n+2, n])    
        for j in range(1,n):
            self._elements_list.append([j, n+1+j*2, n+(j*2)+2, n+1+(j-1)*2])        
        for i in range(1,n):
            j=0
            self._elements_list.append([n+2+(2*n+1)*(i-1) + j*2, n+1+(2*n+1)*i + j*2, n+2+(2*n+1)*i + j*2, n+(2*n+1)*i + j*2])
            for j in range(1,n):
                self._elements_list.append([n+2+(2*n+1)*(i-1) + j*2, n+1+(2*n+1)*i + j*2, n+2+(2*n+1)*i + j*2, n+1+(2*n+1)*i + (j-1)*2])

        self._neighbours.append([1,n])
        for j in range(1,n-1):
            self._neighbours.append([j+1, j+n, j-1])
        self._neighbours.append([n-2, 2*n-1])

        for i in range(1, n-1):
            self._neighbours.append([n*i+1, n*(i+1), n*(i-1)])
            for j in range(1, n-1):
                self._neighbours.append([n*i+j+1, n*(i+1)+j, n*i+j-1, n*(i-1)+j])
            self._neighbours.append([n*(i+2)-1, n*(i+1)-2, n*i-1])

        self._neighbours.append([n*n-n+1, n*(n-2)])
        for j in range(1,n-1):
            self._neighbours.append([n*(n-1)+j-1, n*(n-2)+j, n*(n-1)+j+1])
        self._neighbours.append([n*n-2, n*(n-1)-1])

        for j in range(n):
            self._owner_list.append(j)
            self._neighbour_list.append(-1)
            
        for i in range(n-1):
            self._owner_list.append(-1)
            self._neighbour_list.append(i*n)
            for j in range(n-1):
                self._owner_list.append(i*n + j)
                self._neighbour_list.append(j+1 + i*n)
                self._owner_list.append((i+1)*n+j)
                self._neighbour_list.append(i*n+j)
            self._owner_list.append((i+1)*n-1)
            self._neighbour_list.append(-1)
            self._owner_list.append((i+2)*n-1)
            self._neighbour_list.append((i+1)*n-1)

        self._owner_list.append(-1)
        self._neighbour_list.append(n*(n-1))
        for j in range(n-1):
            self._owner_list.append(n*(n-1) + j)
            self._neighbour_list.append(n*(n-1) + j + 1)
            self._owner_list.append(-1)
            self._neighbour_list.append(n*(n-1) + j)
        self._owner_list.append(n*n-1)
        self._neighbour_list.append(-1)
        self._owner_list.append(-1)
        self._neighbour_list.append(n*n-1)

        self._owner_list = np.array(self._owner_list)
        self._neighbour_list = np.array(self._neighbour_list)

        #print(self._points_list)
        #print(self._faces_list)
        #print(self._elements_list)

    def genSquareMeshTetra(self, L, n):
        
        self._n = n
        x = np.linspace(0,L, n+1)
        X,Y = np.meshgrid(x,x)
        X = X.flatten()
        Y = Y.flatten()
        self._points_list = np.array([X,Y])

        for j in range(n):
            self._faces_list.append([j+1, j])
        for i in range(n):
            self._faces_list.append([(i+1)*(n+1), i*(n+1)])
            for j in range(n):
                self._faces_list.append([j+1 + (i+1)*(n+1), j+1 + i*(n+1)])
                self._faces_list.append([j+1 + (i+1)*(n+1), j + (i+1)*(n+1)])
                self._faces_list.append([j+1 + i*(n+1), (n+1)*(i+1) + j])
        self._faces_list = np.array(self._faces_list)
        
        self._elements_list.append([0, n+3, n])
        self._elements_list.append([n+1, n+2, n+3])   
        for j in range(1,n):
            self._elements_list.append([j, n+3+j*3, n+1+((j-1)*3)])   
            self._elements_list.append([n+1+j*3, n+2+j*3, n+3+j*3])
       
        for i in range(1,n):
            j=0
            self._elements_list.append([n+2+(3*n+1)*(i-1) + j*3, n+3+(3*n+1)*i + j*3, n+(3*n+1)*i + j*3])
            self._elements_list.append([n+1 + (n*3+1)*i + 3*j, n+2 + (n*3+1)*i + 3*j, n+3 + (n*3+1)*i + 3*j])
            for j in range(1,n):
                self._elements_list.append([n+2+(3*n+1)*(i-1) + j*3, n+3+(3*n+1)*i + j*3, n+1+(3*n+1)*i + (j-1)*3])
                self._elements_list.append([n+1 + (n*3+1)*i + 3*j, n+2 + (n*3+1)*i + 3*j, n+3 + (n*3+1)*i + 3*j])


        self._neighbours.append([1])
        for j in range(1,2*n-1):
            if j%2 == 1:
                self._neighbours.append([j+1, j+2*n-1, j-1])
            elif j%2 == 0:
                self._neighbours.append([j+1, j-1])
        self._neighbours.append([4*n-2, 2*n-2])

        for i in range(1,n-1):
            self._neighbours.append([2*n*i+1, 2*n*(i-1)+1])
            for j in range(1,2*n-1):
                if j%2 == 1:
                    self._neighbours.append([2*n*i+j+1, 2*n*(i+1)+j-1, 2*n*i+j-1])
                elif j%2 == 0:
                    self._neighbours.append([2*n*i+j+1, 2*n*i+j-1, 2*n*(i-1)+j+1])
            self._neighbours.append([2*n*(i+2)-2, 2*n*(i+1)-2])

        self._neighbours.append([2*n*(n-2)+1, 2*n*(n-1)+1])
        for j in range(1,2*n-1):
            if j%2 == 1:
                self._neighbours.append([2*n*(n-1)+j+1, 2*n*(n-1)+j-1])
            elif j%2 == 0:
                self._neighbours.append([2*n*(n-1)+j+1, 2*n*(n-1)+j-1, 2*n*(n-2)+j+1])
        self._neighbours.append([2*n*n-2])

        for j in range(n):
            self._owner_list.append(j*2)
            self._neighbour_list.append(-1)

        
        for i in range(n-1):    
            self._owner_list.append(-1)
            self._neighbour_list.append(i*2*n)
            self._owner_list.append(1 + i*2*n)
            self._neighbour_list.append(2 + i*2*n)
            self._owner_list.append(2*n*(i+1))
            self._neighbour_list.append(1 + i*2*n)
            self._owner_list.append(1 + i*2*n)
            self._neighbour_list.append(i*2*n)
            for j in range(1,n-1):
                pass
                self._owner_list.append(j*2+1 + i*2*n)
                self._neighbour_list.append(j*2+2 + i*2*n)
                self._owner_list.append(2*j+2*n*(i+1))
                self._neighbour_list.append(j*2+1 + i*2*n)
                self._owner_list.append(j*2+1 + i*2*n)
                self._neighbour_list.append(j*2 + i*2*n)
            self._owner_list.append(2*n-1 + 2*n*i)
            self._neighbour_list.append(-1)
            self._owner_list.append(2*n-2 + 2*(i+1)*n)
            self._neighbour_list.append(2*n-1 + 2*n*i)
            self._owner_list.append(2*n-1 + 2*n*i)
            self._neighbour_list.append(2*n-2 + 2*n*i) 
        self._owner_list.append(-1)
        self._neighbour_list.append(2*n*(n-1))
        self._owner_list.append(2*n*(n-1)+1)
        self._neighbour_list.append(2*n*(n-1)+2)
        self._owner_list.append(-1)
        self._neighbour_list.append(2*n*(n-1)+1)
        self._owner_list.append(2*n*(n-1)+1)
        self._neighbour_list.append(2*n*(n-1))
        for j in range(1,n-1):
            pass
            self._owner_list.append(j*2+1 + (n-1)*2*n)
            self._neighbour_list.append(j*2+2 + (n-1)*2*n)
            self._owner_list.append(-1)
            self._neighbour_list.append(j*2+1 + (n-1)*2*n)
            self._owner_list.append(j*2+1 + (n-1)*2*n)
            self._neighbour_list.append(j*2 + (n-1)*2*n)
        self._owner_list.append(2*n-1 + 2*n*(n-1))
        self._neighbour_list.append(-1)
        self._owner_list.append(-1)
        self._neighbour_list.append(2*n-1 + 2*n*(n-1))
        self._owner_list.append(2*n-1 + 2*n*(n-1))
        self._neighbour_list.append(2*n-2 + 2*n*(n-1))

        self._owner_list = np.array(self._owner_list)
        self._neighbour_list = np.array(self._neighbour_list)
        
        #print(self._points_list)
        #print(self._faces_list)
        #print(self._elements_list)

    def computeGeometricCenter(self):
        sum_coord = np.zeros((2,))
        for element in self._elements_list:
            used_points = []
            sum_coord.fill(0)
            for face in element:
                points = self._faces_list[face]
                for point in points:
                    if point not in used_points:
                        #print(point)
                        used_points.append(point)
                        coord = self._points_list[:,point]
                        sum_coord = sum_coord + coord
            self._geometricCenterVolume_list.append(sum_coord / len(used_points))

    def computeCentroid(self):
        sum_coord = np.zeros((2,))
        weightedCenter = np.zeros((2,))
        for element, gc in zip(self._elements_list, self._geometricCenterVolume_list):
            sum_coord.fill(0)
            Stotal = 0
            weightedCenter.fill(0)
            for face in element:
                points = self._faces_list[face]
                point2 = self._points_list[:,points[0]]
                point3 = self._points_list[:,points[1]]
                sum_coord = point2 + point3 + gc
                
                localCenter = sum_coord / 3
                S = np.linalg.norm(0.5 * np.cross(point2 - gc, point3 - gc))
                Stotal = Stotal + S
                weightedCenter = weightedCenter + S * localCenter

            self._centroidVolume_list.append(weightedCenter / Stotal)
            self._area_list.append(Stotal)    

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

    def computeGeoDiff(self):
        for i, face in enumerate(self._faces_list):
            owner = self._owner_list[i]
            neighbour = self._neighbour_list[i]
            cf = self._centroidFace_list[i]
            geomDiff= [-1, -1]
            if owner != -1:
                owner_c = self._centroidVolume_list[owner]
                diff = np.linalg.norm(owner_c - cf)
                geomDiff[0] = diff

            if neighbour != -1:
                neighbour_c = self._centroidVolume_list[neighbour]
                diff = np.linalg.norm(neighbour_c - cf)
                geomDiff[1] = diff
            
            if owner != -1 and neighbour != -1:
                
                owner_c = self._centroidVolume_list[owner]
                neighbour_c = self._centroidVolume_list[neighbour]
                self._dCF_list.append(np.linalg.norm(owner_c - neighbour_c))
                self._e.append((neighbour_c - owner_c) / self._dCF_list[-1])
                self._E.append(self._faceNorm_list[i]**2 / np.dot(self._e[-1], self._faceNormal_list[i] * self._faceNorm_list[i]) * self._e[-1])
            
            elif owner != -1:
                
                owner_c = self._centroidVolume_list[owner]
                diff = np.linalg.norm(owner_c - cf)
                self._dCF_list.append(diff)
                self._e.append((cf- owner_c) / self._dCF_list[-1])
                self._E.append(self._faceNorm_list[i]**2 / np.dot(self._e[-1], self._faceNormal_list[i] * self._faceNorm_list[i]) * self._e[-1])
                
            elif neighbour != -1:
                neighbour_c = self._centroidVolume_list[neighbour]
                diff = np.linalg.norm(neighbour_c - cf)
                self._dCF_list.append(diff)
                self._e.append((neighbour_c - cf) / self._dCF_list[-1])
                self._E.append(self._faceNorm_list[i]**2 / np.dot(self._e[-1], self._faceNormal_list[i] * self._faceNorm_list[i]) * self._e[-1])

            self._geomDiff_list.append(geomDiff) # might

        self._T = []
        for n, S, E in zip(self._faceNormal_list, self._faceNorm_list, self._E):
            self._T.append(n*S - E)
        
        self._geomDiff_list = np.array(self._geomDiff_list)
        self._e = np.array(self._e)
        self._dCF_list = np.array(self._dCF_list)
        self._E = np.array(self._E)
        self._T = np.array(self._T)
        
    def reorderFaces(self, type):
        n = self._n
        index = None
        if type == "hexa":
            patch1 = np.linspace(0, n-1, n, dtype=np.int32)
            patch2 = np.linspace(n, n+(2*n+1)*(n-1), n, dtype=np.int32)
            patch3 = np.linspace(n+2*n-1, n+2*n-1+(2*n+1)*(n-1), n, dtype=np.int32)
            patch4 = np.linspace(n*n*2+2*n-1-2*(n-1), n*n*2+2*n-1, n, dtype=np.int32)
            index = np.concatenate((patch1, patch2, patch3, patch4), dtype=np.int32)
            inner = [i for i in range(n*n*2 +2*n) if i not in index]
            index = np.concatenate((inner, index), dtype=np.int32)
            
        elif type == "tetra":
            patch1 = np.linspace(0, n-1, n, dtype=np.int32)
            patch2 = np.linspace(n, n + (n-1)*(n*3+1), n, dtype=np.int32)
            patch3 = np.linspace(n+3*n-2, n+3*n-2+(n-1)*(n*3+1), n, dtype=np.int32)
            patch4 = np.linspace(n*n*3+2*n-2 - (n-1)*3, n*n*3+2*n-2, n, dtype=np.int32)
            index = np.concatenate((patch1, patch2, patch3, patch4), dtype=np.int32)
            inner = [i for i in range(n*n*3 +2*n) if i not in index]
            index = np.concatenate((inner, index), dtype=np.int32)
        
        self._faces_list = self._faces_list[index.astype(int)]
        self._faceNorm_list = self._faceNorm_list[index.astype(int)]
        self._owner_list = self._owner_list[index.astype(int)]
        self._neighbour_list = self._neighbour_list[index.astype(int)]
        self._centroidFace_list = self._centroidFace_list[index.astype(int)]
        self._faceNormal_list = self._faceNormal_list[index.astype(int)]
        self._geomDiff_list = self._geomDiff_list[index.astype(int)]
        self._dCF_list = self._dCF_list[index.astype(int)]
        self._E = self._E[index.astype(int)]
        self._e = self._e[index.astype(int)]
        self._T = self._T[index.astype(int)]


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


