import gmsh
import numpy as np
import subprocess
from fenics import *

file = np.loadtxt("contorno.txt")

gmsh.initialize()

contorno = []
for pt in file:
    contorno.append(gmsh.model.geo.addPoint(pt[0], pt[1], pt[2], 4))

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

input_file = 'malha.msh'
output_file = 'malha.xml'

command = ['dolfin-convert', 'malha.msh', 'malha.xml']

subprocess.run(command, check = True)

mesh = Mesh("malha.xml")

facet_function = MeshFunction("size_t", mesh, "malha_facet_region.xml")

coordinates = mesh.coordinates()

scaling_factor = 0.01

coordinates[:] *= scaling_factor

mesh.bounding_box_tree().build(mesh)

File("malha.xml") << mesh

File("malha_facet_region.xml") << facet_function