import numpy as np
import os
import shutil

# class to write results in legacy vtk format
class VTKWRITER:
    # initialize writer with mesh and output name
    def __init__(self, mesh, name):
        self._mesh = mesh
        self._numPoints = self._mesh._points_list.shape[1]
        
        script_dir = os.path.dirname(__file__)
        rel_path = os.path.join(name, name)
        self._abs_file_path = os.path.join(script_dir, rel_path)
        self._name = name
        if name not in os.listdir():
            os.mkdir(os.path.join(script_dir, name))
        else:
            shutil.rmtree(os.path.join(script_dir, name))
            os.mkdir(os.path.join(script_dir, name))
        
        self._numElements = self._mesh._numElements
        
    # write geometry to vtk file
    def writeGeometry(self, timestep):
        numPoints = self._numPoints
        f = open(self._abs_file_path+"_"+str(timestep)+".vtk", "w")
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Sim data\n")
        f.write("ASCII\n\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {numPoints} double\n")       
        for i in range(numPoints):
            f.write(f"{self._mesh._points_list[0,i]} {self._mesh._points_list[1,i]} 0.0\n")
        
        
        center = self._mesh._centroidVolume_list
        elements = [[] for _ in range(self._numElements)]
        
        owner = self._mesh._owner_list
        neighbour = self._mesh._neighbour_list
        for i, face in enumerate(self._mesh._faces_list):
            if face[0] not in elements[owner[i]]:
                elements[owner[i]].append((face[0]))
            if face[1] not in elements[owner[i]]:
                elements[owner[i]].append((face[1]))
            
            if i < self._mesh._bndStart:
                #print(face)
                if face[0] not in elements[neighbour[i]]:
                    elements[neighbour[i]].append((face[0]))
                if face[1] not in elements[neighbour[i]]:
                    elements[neighbour[i]].append((face[1]))

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

    # write header before storing simulation data (not geometry!)
    def writeDataHeader(self, timestep):
        f = open(self._abs_file_path+"_"+str(timestep)+".vtk", "a")
        f.write(f"\nCELL_DATA {self._numElements}\n")

    # write scalar
    def writeScalar(self, scalar, timestep, name):
        #self.writeGeometry(timestep)
        
        f = open(self._abs_file_path+"_"+str(timestep)+".vtk", "a")
        #f.write(f"\nCELL_DATA {self._numElements}\n")
        f.write("SCALARS " + name + " double\n")
        f.write("LOOKUP_TABLE default\n")
        for t in scalar[:-1]:
            f.write(f"{t}\n")
        f.write(f"{scalar[-1]}")

        f.close()

    # write vector
    def writeVector(self, vector, timestep, name):
        f = open(self._abs_file_path+"_"+str(timestep)+".vtk", "a")
        
        f.write("\nVECTORS " + name + " double\n")
        for t in vector[:-1]:
            f.write(f"{t[0]} {t[1]} 0\n")
        t = vector[-1]
        f.write(f"{t[0]} {t[1]} 0")

        f.close()

    # write series json file to import time values
    def writeSeries(self, time):
        f = open(self._abs_file_path + ".vtk.series", "w")
        f.write("{\n")
        f.write("\t\"file-series-version\" : \"1.0\",\n")
        f.write("\t\"files\" : [\n")

        files = (os.listdir(os.path.join(os.path.dirname(__file__))+"/" + self._name))
        files = [f for f in files if ".series" not in f]
        files = sorted(files, key = lambda x:int(x[len(self._name)+1:-4]))
        
        for file, t in zip(files[:-1], time[:-1]):
            f.write("\t\t{ \"name\" : \"" + file + "\", \"time\" : " + str(round(t, 2)) + " },\n")

        f.write("\t\t{ \"name\" : \"" + files[-1] + "\", \"time\" : " + str(round(time[-1], 2)) + " }\n")

        f.write("\t]\n")
        f.write("}")