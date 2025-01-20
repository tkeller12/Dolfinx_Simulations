import gmsh
import math
import sys

import numpy as np

gap_width = 0.1
gap_length = 1
sample_loop_radius = 1.0
return_loop_radius = 2.0

Lc1 = 0.2

gmsh.initialize()

gmsh.model.add('lgr')

factory = gmsh.model.geo

y1 = np.sqrt(sample_loop_radius**2 - (gap_width/2)**2.)
y2 = sample_loop_radius + gap_length + return_loop_radius - np.sqrt(return_loop_radius**2 - (gap_width/2)**2.)
x = gap_width/2

factory.addPoint(0, 0, 0, Lc1, 1) # (x, y, z, mesh_size, tag) center of sample loop
factory.addPoint(-x, y1, 0, Lc1, 2) # exact
factory.addPoint(x, y1, 0, Lc1, 3) # exact

#dist = np.sqrt(x**2 + y1**2)
#print('calc dist:', dist)
#print('radius:', sample_loop_radius)

factory.addPoint(0, sample_loop_radius + gap_length + return_loop_radius, 0, Lc1, 4) # center of return loop
factory.addPoint(x, y2, 0, Lc1, 5) # exact
factory.addPoint(-x, y2, 0, Lc1, 6) # exact

factory.addPoint(0, -sample_loop_radius, 0, Lc1, 7)
factory.addPoint(0, sample_loop_radius+gap_length+2*return_loop_radius, 0, Lc1, 8)

#factory.addLine(1, 2, 1) # arc
#factory.addCircleArc(2, 1, 3, 1) # (start, center, end, tag)
factory.addCircleArc(2, 1, 7, 1) # (start, center, end, tag)
factory.addCircleArc(7, 1, 3, 2) # (start, center, end, tag)
factory.addLine(3, 5, 3) # (start, end, tag)
factory.addCircleArc(5, 4, 8, 4)
factory.addCircleArc(8, 4, 6, 5)
factory.addLine(6, 2, 6)

factory.addCurveLoop([1, 2, 3, 4, 5, 6], 5)
factory.addPlaneSurface([5], 6)

factory.synchronize()

gmsh.model.mesh.generate(2)

gmsh.write("lgr_test.msh")

gmsh.finalize()

