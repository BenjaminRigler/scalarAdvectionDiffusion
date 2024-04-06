The following project solves the 2d advection diffusion problem on unstructured grids supporting quadrilaterals and triangle elements. The convection term can be discretized using HO schemes. For time stepping the backward Euler method is used. The gradient field is computed using the Divergence theorem.
The mesh and boundary conditions are defined using the openFoam structure, therefore a folder "constant/polyMesh" containing the files "boundary", "faces", "neighbour", "owner" and "points" is required. To define the boundary conditions a file "phi" in the "0" directory is needed. Results are written in the legacy vtk format.
The workflow is presented in the file swirl.py for a convection problem advecting a rectange in a rotating velocity field. The required boundary and mesh files are included in the repository. The results are shown next:
![](https://github.com/BenjaminRigler/scalarAdvectionDiffusion/blob/main/swirl.gif)
The comparison of meshes using quadrilateral and triangle elements are presented next for a simple diffusion problem:
![](https://github.com/BenjaminRigler/scalarAdvectionDiffusion/blob/main/diff_meshComp.png)
![](https://github.com/BenjaminRigler/scalarAdvectionDiffusion/blob/main/diff_lineplot.png)
Note that for advection dominated problems the code has convergence issues except for orthogonal cartesian grids.
