import numpy as np

class VTKWRITER:
    def __init__(self, mesh, name):
        self._mesh = mesh
        self._numPoints = self._mesh._points_list.shape[1]
        self._name = name
        self._numElements = max(max(self._mesh._owner_list), max(self._mesh._neighbour_list))+1 # consider only using max(owner) for openfoam meshes
        
    
    def writeGeometry(self, timestep):
        name = self._name
        numPoints = self._numPoints
        f = open(name+str(timestep)+".vtk", "w")
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Sim data\n")
        f.write("ASCII\n\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {numPoints} double\n")       
        for i in range(numPoints):
            f.write(f"{self._mesh._points_list[0,i]} {self._mesh._points_list[1,i]} 0.0\n")
        
        
        center = self._mesh._centroidVolume_list
        elements = [[] for _ in range(self._numElements)]
        for face, owner, neighbour in zip(self._mesh._faces_list, self._mesh._owner_list, self._mesh._neighbour_list):
            if owner != -1:
                if face[0] not in elements[owner]:
                    elements[owner].append((face[0]))
                if face[1] not in elements[owner]:
                    elements[owner].append((face[1]))
            if neighbour != -1:
                if face[0] not in elements[neighbour]:
                    elements[neighbour].append((face[0]))
                if face[1] not in elements[neighbour]:
                    elements[neighbour].append((face[1]))

        f.write(f"\nCELLS {self._numElements} {np.sum([len(e) for e in elements]) + self._numElements}\n")
        for i, c in enumerate(center):
            phi = []
            points = elements[i]
            for p in points:
                px = self._mesh._points_list[0,p]
                py = self._mesh._points_list[1,p]
                phi.append(np.arctan2(px-c[0], py-c[1]))

            sortedPoints = np.array(points)[np.flip(np.argsort(phi))]
            f.write(f"{len(sortedPoints)} ")
            for p in sortedPoints[:-1]:
                f.write(f"{p} ")
            f.write(f"{sortedPoints[-1]}\n")
            
        f.write(f"\nCELL_TYPES {self._numElements}\n")    
        for i in range(self._numElements):
            order = len(elements[i])
            if order == 4:
                f.write("9\n")
            elif order == 3:
                f.write("5\n")

        
        f.close()

    def writeScalar(self, scalar, timestep):
        self.writeGeometry(timestep)

        f = open(self._name+str(timestep)+".vtk", "a")
        f.write(f"\nCELL_DATA {self._numElements}\n")
        f.write("SCALARS T double\n")
        f.write("LOOKUP_TABLE default\n")
        for t in scalar[:-1]:
            f.write(f"{t}\n")
        f.write(f"{scalar[-1]}")

        f.close()

    def writeVector(self, vector, timestep):
        f = open(self._name+str(timestep)+".vtk", "a")
        
        f.write("\nVECTORS gradT double\n")
        for t in vector[:-1]:
            f.write(f"{t[0]} {t[1]} 0\n")
        t = vector[-1]
        f.write(f"{t[0]} {t[1]} 0")

        f.close()