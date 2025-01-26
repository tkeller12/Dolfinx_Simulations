import gmsh
import math
import sys

import numpy as np

gap_width = 0.65e-3
gap_length = 1.6e-3
sample_loop_radius = 2.6e-3
return_loop_radius = 3.15e-3
height = 8e-3

void_thickness = 1e-3
void_height = 5e-3
void_radius = sample_loop_radius+gap_length+2*return_loop_radius+void_thickness


Lc0 = 0.10 * sample_loop_radius

Lc1 = 0.5 * sample_loop_radius
Lc2 = 1.0 * sample_loop_radius
Lc3 = 1.5 * sample_loop_radius

lgr_extrude_divisions = 8
void_extrude_divisions = 1

gmsh.initialize()

gmsh.model.add('lgr')

#factory = gmsh.model.geo
factory = gmsh.model.occ

y1 = np.sqrt(sample_loop_radius**2 - (gap_width/2)**2.)
y2 = sample_loop_radius + gap_length + return_loop_radius - np.sqrt(return_loop_radius**2 - (gap_width/2)**2.)

y3 = (y1 + y2) / 2 # midpoint

y4 = (y1 + y2) / 2 - gap_length / 4

y5 = (y1 + y2) / 2 + gap_length / 4

x = gap_width/2

### ADD POINTS ###
p_center = factory.addPoint(0, 0, -height/2, Lc1, 1) # (x, y, z, mesh_size, tag) center of sample loop
factory.addPoint(x, y1, -height/2, Lc1, 2) # exact

factory.addPoint(x, y2, -height/2, Lc2, 3) # exact
factory.addPoint(0, sample_loop_radius + gap_length + return_loop_radius, -height/2, Lc2, 4) # center of return loop
factory.addPoint(0, sample_loop_radius + gap_length + 2*return_loop_radius, -height/2, Lc2, 5) # center of return loop
factory.addPoint(-x, y2, -height/2, Lc2, 6) # exact
factory.addPoint(-x, y1, -height/2, Lc1, 7) # exact

factory.addPoint(-x, -y1, -height/2, Lc1, 8) # exact
factory.addPoint(-x, -y2, -height/2, Lc2, 9) # exact

factory.addPoint(0, -1*sample_loop_radius - gap_length - return_loop_radius, -height/2, Lc2, 10) # center of return loop
factory.addPoint(0, -1*sample_loop_radius - gap_length - 2*return_loop_radius, -height/2, Lc2, 11) # center of return loop
factory.addPoint(x, -y2, -height/2, Lc2, 12) # exact
factory.addPoint(x, -y1, -height/2, Lc1, 13) # exact

p_m1 = factory.addPoint(x, y3, -height/2, Lc0, 101) # exact
p_m2 = factory.addPoint(-x, y3, -height/2, Lc0, 102) # exact
p_m3 = factory.addPoint(-x, -y3, -height/2, Lc0, 103) # exact
p_m4 = factory.addPoint(x, -y3, -height/2, Lc0, 104) # exact

p_y4 = factory.addPoint(x, y4, -height/2, Lc0, 105) # exact
p_y5 = factory.addPoint(x, y5, -height/2, Lc0, 106) # exact

### ADD LINES ###
factory.addCircleArc(13, 1, 2, 1) # (start, center, end, tag)
factory.addLine(2, p_y4, 2) # (start, end, tag)
factory.addLine(p_y4, p_y5, 3) # (start, end, tag)
factory.addLine(p_y5, 3, 4) # (start, end, tag)
factory.addCircleArc(3, 4, 5, 5) # (start, center, end, tag)
factory.addCircleArc(5, 4, 6, 6) # (start, center, end, tag)
factory.addLine(6, p_m2, 7) # (start, end, tag)
factory.addLine(p_m2, 7, 8) # (start, end, tag)
factory.addCircleArc(7, 1, 8, 9) # (start, center, end, tag)
factory.addLine(8, 9, 10) # (start, end, tag)
factory.addCircleArc(9, 10, 11, 11) # (start, center, end, tag)
factory.addCircleArc(11, 10, 12, 12) # (start, center, end, tag)
factory.addLine(12, 13, 13) # (start, end, tag)

factory.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 14)
curve = factory.addPlaneSurface([14], 15)

resonator = factory.extrude([(2, curve)], 0, 0, height, [lgr_extrude_divisions]) # ((dimTags) dx, dy, dz, [subdiv1,...]) 
factory.synchronize()




#gmsh.model.addPhysicalGroup(3, [1], name = 'Resonator')#, 1) # need to add physcial groups
#gmsh.model.addPhysicalGroup(3, [4], name = 'Resonator')#, 1) # need to add physcial groups

#gmsh.model.mesh.setAlgorithm(3, 1, 11)
#gmsh.model.mesh.Algorithm3D(3, 1, 10)

#gmsh.option.setNumber("Mesh.Algorithm3D", 10) #HXT
#gmsh.option.setNumber("Mesh.Algorithm3D", 7) #MMG3D, horrible option

#gmsh.option.setNumber("Mesh.Algorithm3D", 1) #Delaunay, default, horrible option

#gmsh.option.setNumber("Mesh.Algorithm3D", 9) #R-tree, mesh looks good, good option
#gmsh.option.setNumber("Mesh.Algorithm3D", 4) #Frontal, mesh looks good, good option
gmsh.model.mesh.generate(3)
#gmsh.model.mesh.refine()

gmsh.write("lgr_2d_test3.msh")

#print(void_bottom)
#print(void_bottom[0])
#print(out1)
#print(out2)

gmsh.fltk.run()

# Finalize the GMSH API
gmsh.finalize()


