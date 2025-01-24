import gmsh
import math
import sys

import numpy as np

gap_width = 0.1
gap_length = 1
sample_loop_radius = 1.0
return_loop_radius = 2.0
height = 1

void_thickness = 0.1
void_height = 1
void_radius = sample_loop_radius+gap_length+2*return_loop_radius+void_thickness


Lc1 = 0.5

gmsh.initialize()

gmsh.model.add('lgr')

factory = gmsh.model.geo

y1 = np.sqrt(sample_loop_radius**2 - (gap_width/2)**2.)
y2 = sample_loop_radius + gap_length + return_loop_radius - np.sqrt(return_loop_radius**2 - (gap_width/2)**2.)
x = gap_width/2

### ADD POINTS ###
factory.addPoint(0, 0, -height/2, Lc1, 1) # (x, y, z, mesh_size, tag) center of sample loop
factory.addPoint(x, y1, -height/2, Lc1, 2) # exact
factory.addPoint(x, y2, -height/2, Lc1, 3) # exact
factory.addPoint(0, sample_loop_radius + gap_length + return_loop_radius, -height/2, Lc1, 4) # center of return loop
factory.addPoint(0, sample_loop_radius + gap_length + 2*return_loop_radius, -height/2, Lc1, 5) # center of return loop
factory.addPoint(-x, y2, -height/2, Lc1, 6) # exact
factory.addPoint(-x, y1, -height/2, Lc1, 7) # exact

factory.addPoint(-x, -y1, -height/2, Lc1, 8) # exact
factory.addPoint(-x, -y2, -height/2, Lc1, 9) # exact

factory.addPoint(0, -1*sample_loop_radius - gap_length - return_loop_radius, -height/2, Lc1, 10) # center of return loop
factory.addPoint(0, -1*sample_loop_radius - gap_length - 2*return_loop_radius, -height/2, Lc1, 11) # center of return loop
factory.addPoint(x, -y2, -height/2, Lc1, 12) # exact
factory.addPoint(x, -y1, -height/2, Lc1, 13) # exact


### ADD LINES ###
factory.addCircleArc(13, 1, 2, 1) # (start, center, end, tag)
factory.addLine(2, 3, 2) # (start, end, tag)
factory.addCircleArc(3, 4, 5, 3) # (start, center, end, tag)
factory.addCircleArc(5, 4, 6, 4) # (start, center, end, tag)
factory.addLine(6, 7, 5) # (start, end, tag)
factory.addCircleArc(7, 1, 8, 6) # (start, center, end, tag)
factory.addLine(8, 9, 7) # (start, end, tag)
factory.addCircleArc(9, 10, 11, 8) # (start, center, end, tag)
factory.addCircleArc(11, 10, 12, 9) # (start, center, end, tag)
factory.addLine(12, 13, 10) # (start, end, tag)

factory.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)
curve = factory.addPlaneSurface([11], 12)

resonator = factory.extrude([(2, curve)], 0, 0, height) # ((dimTags) dx, dy, dz) need extrusion for mesh

### ADD TOP VOID ###
p0 = factory.addPoint(0,0,height/2)
p1 = factory.addPoint(void_radius,0,height/2)
p2 = factory.addPoint(0,void_radius,height/2)
p3 = factory.addPoint(-void_radius,0,height/2)
p4 = factory.addPoint(0,-void_radius,height/2)

arc0 = factory.addCircleArc(p1, p0, p2)
arc1 = factory.addCircleArc(p2, p0, p3)
arc2 = factory.addCircleArc(p3, p0, p4)
arc3 = factory.addCircleArc(p4, p0, p1)

curve_loop = factory.addCurveLoop([arc0, arc1, arc2, arc3])
curve_void = factory.addPlaneSurface([curve_loop])

void_top = factory.extrude([(2, curve_void)], 0, 0, void_height) # ((dimTags) dx, dy, dz) need extrusion for mesh

### ADD Bottom VOID ###
p0 = factory.addPoint(0,0,-height/2)
p1 = factory.addPoint(void_radius,0,-height/2)
p2 = factory.addPoint(0,void_radius,-height/2)
p3 = factory.addPoint(-void_radius,0,-height/2)
p4 = factory.addPoint(0,-void_radius,-height/2)

arc0 = factory.addCircleArc(p1, p0, p2)
arc1 = factory.addCircleArc(p2, p0, p3)
arc2 = factory.addCircleArc(p3, p0, p4)
arc3 = factory.addCircleArc(p4, p0, p1)

curve_loop = factory.addCurveLoop([arc0, arc1, arc2, arc3])
curve_void = factory.addPlaneSurface([curve_loop])

void_bottom = factory.extrude([(2, curve_void)], 0, 0, -void_height) # ((dimTags) dx, dy, dz) need extrusion for mesh

factory.synchronize()
#gmsh.model.addPhysicalGroup(3, [curve,void_top,void_bottom], name = 'LGR Resonator')#, 1) # need to add physcial groups

#gmsh.model.addPhysicalGroup(3, [resonator, void_top, void_bottom], name = 'LGR Resonator')#, 1) # need to add physcial groups
#gmsh.model.addPhysicalGroup(3, [1,2,3], name = 'LGR Resonator')#, 1) # need to add physcial groups
gmsh.model.addPhysicalGroup(3, [1], name = 'LGR Resonator')#, 1) # need to add physcial groups
gmsh.model.addPhysicalGroup(3, [2], name = 'Top Void')#, 1) # need to add physcial groups
gmsh.model.addPhysicalGroup(3, [3], name = 'Bottom Void')#, 1) # need to add physcial groups
gmsh.model.mesh.generate(3)

gmsh.write("mesh/lgr_3d_test3.msh")

print(void_bottom)
print(void_bottom[0])

gmsh.finalize()

