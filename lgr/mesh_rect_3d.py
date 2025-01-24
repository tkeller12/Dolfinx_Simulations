import gmsh
import math
import sys

import numpy as np


a = 1.0
b = 0.5
d = 0.75

Lc1 = 0.2

gmsh.initialize()

gmsh.model.add('rect')

factory = gmsh.model.geo

x = a/2
y = b/2

factory.addPoint(-x, -y, 0, Lc1, 1) # (x, y, z, mesh_size, tag) 
factory.addPoint(x, -y, 0, Lc1, 2) # exact
factory.addPoint(x, y, 0, Lc1, 3) # exact
factory.addPoint(-x, y, 0, Lc1, 4) # exact


factory.addLine(1, 2, 1) # (start, end, tag)
factory.addLine(2, 3, 2)
factory.addLine(3, 4, 3)
factory.addLine(4, 1, 4)

factory.addCurveLoop([1, 2, 3, 4], 5)
surface = factory.addPlaneSurface([5], 6)

factory.extrude([(2, surface)], 0, 0, d, numElements = [10]) # ((dimTags) dx, dy, dz) need extrusion for mesh
gmsh.model.addPhysicalGroup(2, [6], 1) # need to add physcial groups

factory.synchronize()

gmsh.model.mesh.generate(3)

gmsh.write("rect_3d_test.msh")

gmsh.finalize()

