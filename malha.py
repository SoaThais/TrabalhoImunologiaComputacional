import gmsh
import numpy as np

file = np.loadtxt("contorno.txt")

gmsh.initialize()

contorno = []
for pt in file:
    contorno.append(gmsh.model.geo.addPoint(pt[0], pt[1], pt[2], 5))

sp_contorno = gmsh.model.geo.addSpline(contorno)
sp_contorno2 = gmsh.model.geo.addSpline([contorno[-1], contorno[0]])

cl_contorno = gmsh.model.geo.addCurveLoop([sp_contorno, sp_contorno2])

gmsh.model.addPhysicalGroup(1, [sp_contorno, sp_contorno2], tag = 10)

cl_list = [cl_contorno]

gmsh.model.geo.synchronize()

surface = gmsh.model.geo.addPlaneSurface(cl_list)
gmsh.model.addPhysicalGroup(2, [surface], tag=20)

gmsh.model.geo.synchronize()

gmsh.write("malha.geo_unrolled")

gmsh.model.mesh.generate(2)

gmsh.option.setNumber("Mesh.MshFileVersion", 2)

gmsh.write("malha.msh")

gmsh.clear()
gmsh.finalize()