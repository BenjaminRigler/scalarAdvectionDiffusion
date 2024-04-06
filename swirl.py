import numpy as np
import Mesh
import Solver

# read mesh
OFmesh = Mesh.Mesh()
OFmesh.readBndFile()
OFmesh.readOpenFOAM()
OFmesh.computeGeometricCenterFaceBased()
OFmesh.computeCentroidFaceBased()
OFmesh.computeFaceValues()
OFmesh.computeGeoDiffFaceBased()

# define rotor velocity field
def initRotorField(x,y):
    vx = y-0.5
    vy = -(x-0.5)
    n = np.linalg.norm([vx,vy], axis=0)
    vx = vx / n
    vy = vy / n
    return [vx, vy]

# init scalar with a rectangle    
def initRectangle(x,y):
    if x >= 0.6 and x <= 0.8 and y >= 0.4 and y <= 0.6:
        return 1
    return 0

# setup solver
solver = Solver.Solver(OFmesh)
solver.setSimParameter(0, 0.005, 10)
solver.setNumPar(0.5, 1e-06, 1000)
solver.setVelocity(initRotorField)
solver.loadMatrix(1, 0)
solver.setOutput("swirl_test", 5)
solver.initField(initRectangle)
solver.runAdvDiff()
