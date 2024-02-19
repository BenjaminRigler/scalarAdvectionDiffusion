import numpy as np
import Mesh
import Diff
import Solver

n = 20
type = "tetra"
numFaces=0
topStart=0 
eastStart=0
westStart=0
bottomStart=0

mesh = Mesh.Mesh()
if type == "hexa":
    mesh.genSquareMeshHexa(1,n)
    mesh.computeGeometricCenter()
    mesh.computeCentroid()
    mesh.computeFaceValues()
    mesh.computeGeoDiff()
    mesh.reorderFaces("hexa")
    numFaces = n*n*2 + 2*n
    topStart = numFaces-n
    eastStart = topStart-n
    westStart = eastStart-n
    bottomStart = westStart-n

elif type == "tetra":
    mesh.genSquareMeshTetra(1,n)
    mesh.computeGeometricCenter()
    mesh.computeCentroid()
    mesh.computeFaceValues()
    mesh.computeGeoDiff()
    mesh.reorderFaces("tetra")
    numFaces = n*n*3 + 2*n
    topStart = numFaces-n
    eastStart = topStart-n
    westStart = eastStart-n
    bottomStart = westStart-n
    
mesh.plotMeshAll()

patch1 = {"startFace": bottomStart,
          "numFaces": n,
          "type": "D",
          "value": 0}

patch2 = {"startFace": westStart,
          "numFaces": n,
          "type": "N",
          "value": 0}

patch3 = {"startFace": eastStart,
          "numFaces": n,
          "type": "D",
          "value": 1}

patch4 = {"startFace": topStart,
          "numFaces": n,
          "type": "N",
          "value": 0}

boundary = [patch1, patch2, patch3, patch4]

solver = Solver.Solver(mesh)
solver.setBoundary(boundary)
solver.setSimParameter(1, 0.2, 1000)
solver.loadMatrix()
solver.initField(lambda x,y: 0)
solver.setVelocity(lambda x,y: [-1,1])
#solver.runSimulation()
solver.simConvection()