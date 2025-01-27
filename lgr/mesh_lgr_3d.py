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


Lc1 = 1.0 * sample_loop_radius
Lc2 = 1.0 * sample_loop_radius
Lc3 = 1.5 * sample_loop_radius

lgr_extrude_divisions = 4
void_extrude_divisions = 4

gmsh.initialize()

gmsh.model.add('lgr')

#factory = gmsh.model.geo
factory = gmsh.model.occ

y1 = np.sqrt(sample_loop_radius**2 - (gap_width/2)**2.)
y2 = sample_loop_radius + gap_length + return_loop_radius - np.sqrt(return_loop_radius**2 - (gap_width/2)**2.)

y3 = (y1 + y2) / 2 # midpoint

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

#p_m1 = factory.addPoint(x, y3, -height/2, Lc1, 101) # exact
#p_m2 = factory.addPoint(-x, y3, -height/2, Lc1, 102) # exact
#p_m3 = factory.addPoint(-x, -y3, -height/2, Lc1, 101) # exact
#p_m4 = factory.addPoint(x, -y3, -height/2, Lc1, 101) # exact

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
#factory.addWire([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11)
curve = factory.addPlaneSurface([11], 12)

#resonator = factory.extrude([(2, curve)], 0, 0, height, heights = [0.2]) # ((dimTags) dx, dy, dz) need extrusion for mesh
#resonator = factory.extrude([(2, curve)], 0, 0, height, [8,2],[0.5,1]) # ((dimTags) dx, dy, dz) need extrusion for mesh
resonator = factory.extrude([(2, curve)], 0, 0, height, [lgr_extrude_divisions]) # ((dimTags) dx, dy, dz, [subdiv1,...]) 
factory.synchronize()

### ADD TOP VOID ###
#p0 = factory.addPoint(0,0,height/2, Lc3)
#p1 = factory.addPoint(void_radius,0,height/2, Lc3)
#p2 = factory.addPoint(0,void_radius,height/2, Lc3)
#p3 = factory.addPoint(-void_radius,0,height/2, Lc3)
#p4 = factory.addPoint(0,-void_radius,height/2, Lc3)

#arc0 = factory.addCircleArc(p1, p0, p2)
#arc1 = factory.addCircleArc(p2, p0, p3)
#arc2 = factory.addCircleArc(p3, p0, p4)
#arc3 = factory.addCircleArc(p4, p0, p1)
c1 = factory.addCircle(0,0,height/2, void_radius)

#curve_loop = factory.addCurveLoop([arc0, arc1, arc2, arc3])
curve_loop = factory.addCurveLoop([c1])
#curve_loop = factory.addWire([arc0, arc1, arc2, arc3])
curve_void = factory.addPlaneSurface([curve_loop])

void_top = factory.extrude([(2, curve_void)], 0, 0, void_height, [], [], recombine = False) # ((dimTags) dx, dy, dz) need extrusion for mesh
print('void_top',void_top)
void_top = void_top[1]

### ADD Bottom VOID ###
#p0 = factory.addPoint(0,0,-height/2, Lc3)
#p1 = factory.addPoint(void_radius,0,-height/2, Lc3)
#p2 = factory.addPoint(0,void_radius,-height/2, Lc3)
#p3 = factory.addPoint(-void_radius,0,-height/2, Lc3)
#p4 = factory.addPoint(0,-void_radius,-height/2, Lc3)

c2 = factory.addCircle(0,0,-height/2, void_radius)
curve_loop2 = factory.addCurveLoop([c2])
curve_void2 = factory.addPlaneSurface([curve_loop2])

#arc0 = factory.addCircleArc(p1, p0, p2)
#arc1 = factory.addCircleArc(p2, p0, p3)
#arc2 = factory.addCircleArc(p3, p0, p4)
#arc3 = factory.addCircleArc(p4, p0, p1)

#curve_loop = factory.addCurveLoop([curve_void])
curve_void = factory.addPlaneSurface([curve_loop2])

void_bottom = factory.extrude([(2, curve_void2)], 0, 0, -void_height) # ((dimTags) dx, dy, dz) need extrusion for mesh



out1 = gmsh.model.occ.fuse([(3,1)],[(3,2)])
print(out1)
#out1 = gmsh.model.occ.fuse([(3,1)],void_top)
out2 = gmsh.model.occ.fuse([(3,4)],[(3,3)])
factory.synchronize()

#gmsh.model.addPhysicalGroup(3, [1,2,3], name = 'LGR Resonator')#, 1) # need to add physcial groups
#gmsh.model.addPhysicalGroup(3, [2], name = 'Top Void')#, 1) # need to add physcial groups
#gmsh.model.addPhysicalGroup(3, [3], name = 'Bottom Void')#, 1) # need to add physcial groups
#gmsh.model.addPhysicalGroup(3, [1,2,3], name = 'Resonator')#, 1) # need to add physcial groups

gmsh.model.addPhysicalGroup(3, [1], name = 'Resonator')#, 1) # need to add physcial groups
#gmsh.model.addPhysicalGroup(3, [4], name = 'Resonator')#, 1) # need to add physcial groups

#gmsh.model.mesh.setAlgorithm(3, 1, 11)
#gmsh.model.mesh.Algorithm3D(3, 1, 10)

#gmsh.option.setNumber("Mesh.Algorithm3D", 10) #HXT
#gmsh.option.setNumber("Mesh.Algorithm3D", 7) #MMG3D, horrible option

#gmsh.option.setNumber("Mesh.Algorithm3D", 1) #Delaunay, default, horrible option

gmsh.option.setNumber("Mesh.Algorithm3D", 9) #R-tree, mesh looks good, good option
#gmsh.option.setNumber("Mesh.Algorithm3D", 4) #Frontal, mesh looks good, good option
gmsh.model.mesh.generate(3)
#gmsh.model.mesh.refine()
#gmsh.model.mesh.recombine()

gmsh.write("mesh/lgr_3d_test3.msh")

try:
    gmsh.fltk.run()
except:
    pass

gmsh.finalize()

