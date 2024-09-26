import gmsh
import numpy as np
import subprocess
from fenics import *

epi_points = np.loadtxt('EPI_B.txt')
vd_points  = np.loadtxt('VD_B.txt')
ve_points  = np.loadtxt('VE_B.txt')

gmsh.initialize()

epi = [] 
for pt in epi_points:
    epi.append(gmsh.model.geo.addPoint(pt[0], pt[1], pt[2], 4))

vd = []
for pt in vd_points:
    vd.append(gmsh.model.geo.addPoint(pt[0], pt[1], pt[2], 4))

ve = []
for pt in ve_points:
    ve.append(gmsh.model.geo.addPoint(pt[0], pt[1], pt[2], 4))

sp_epi = gmsh.model.geo.addSpline(epi)
sp_epi2 = gmsh.model.geo.addSpline([epi[-1], epi[0]])
sp_vd = gmsh.model.geo.addSpline(vd)
sp_vd2 = gmsh.model.geo.addSpline([vd[-1], vd[0]])
sp_ve = gmsh.model.geo.addSpline(ve)
sp_ve2 = gmsh.model.geo.addSpline([ve[-1], ve[0]])

cl_epi = gmsh.model.geo.addCurveLoop([sp_epi, sp_epi2])
gmsh.model.addPhysicalGroup(1, [sp_epi, sp_epi2], tag = 10)

cl_vd = gmsh.model.geo.addCurveLoop([sp_vd, sp_vd2])
gmsh.model.addPhysicalGroup(1, [sp_vd, sp_vd2], tag = 20)

cl_ve = gmsh.model.geo.addCurveLoop([sp_ve, sp_ve2])
gmsh.model.addPhysicalGroup(1, [sp_ve, sp_ve2], tag = 30)

cl_list = [cl_epi, cl_vd, cl_ve]

gmsh.model.geo.synchronize()

surface = gmsh.model.geo.addPlaneSurface(cl_list)
gmsh.model.geo.synchronize()
gmsh.model.addPhysicalGroup(2, [surface], tag = 40)

gmsh.model.geo.synchronize()

gmsh.write("malha_B.geo_unrolled")

gmsh.model.mesh.generate(2)

gmsh.option.setNumber("Mesh.MshFileVersion", 2)

gmsh.write("malha_B.msh")

gmsh.clear()
gmsh.finalize()

input_file = 'malha_B.msh'
output_file = 'malha_B.xml'

command = ['dolfin-convert', 'malha_B.msh', 'malha_B.xml']

subprocess.run(command, check = True)

mesh = Mesh("malha_B.xml")

facet_function = MeshFunction("size_t", mesh, "malha_B_facet_region.xml")

coordinates = mesh.coordinates()

scaling_factor = 0.01

coordinates[:] *= scaling_factor

mesh.bounding_box_tree().build(mesh)

File("malha_B.xml") << mesh

File("malha_B_facet_region.xml") << facet_function