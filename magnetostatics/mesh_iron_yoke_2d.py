import numpy as np

import gmsh

gmsh.initialize()

gmsh.model.add("2D_Rectangle")

factory = gmsh.model.occ

box_size_x = 1.0
box_size_y = 1.0

pole_gap = 0.1
yoke_width = 1.0
yoke_height = 1.0
yoke_thickness = 0.2
pole_radius = 0.1

meshSize = 0.005

insulation_gap = 0.01

DEFAULT_MESH_SIZE = 0.005

def addRectangle(x, y, z, dx, dy, meshSize = DEFAULT_MESH_SIZE):
    p1 = factory.addPoint(x, y, z, meshSize)
    p2 = factory.addPoint(x+dx, y, z, meshSize)
    p3 = factory.addPoint(x+dx, y+dy, z, meshSize)
    p4 = factory.addPoint(x, y+dy, z, meshSize)

    lines = []

    lines.append(factory.addLine(p1,p2))
    lines.append(factory.addLine(p2,p3))
    lines.append(factory.addLine(p3,p4))
    lines.append(factory.addLine(p4,p1))


    loop = factory.addCurveLoop(lines)
    rect = factory.addPlaneSurface([loop])

    return rect


#rect = factory.addRectangle(0.0, 0.0, 0.0, box_size_x, box_size_y)
rect = addRectangle(0.0, 0.0, 0.0, box_size_x, box_size_y)



current = addRectangle(pole_gap/2.0+insulation_gap, pole_radius+insulation_gap, 0.0, 0.2, 0.15)

### Define Yoke
p1 = factory.addPoint(pole_gap/2.0, 0.0, 0.0, meshSize)
p2 = factory.addPoint(yoke_width/2.0, 0.0, 0.0, meshSize)
p3 = factory.addPoint(yoke_width/2.0, yoke_height/2.0, 0.0, meshSize)
p4 = factory.addPoint(0.0, yoke_width/2.0, 0.0, meshSize)
p5 = factory.addPoint(0.0, yoke_width/2.0 - yoke_thickness, 0.0, meshSize)
p6 = factory.addPoint(yoke_width/2.0 - yoke_thickness, yoke_width/2.0 - yoke_thickness, 0.0, meshSize)
p7 = factory.addPoint(yoke_width/2.0 - yoke_thickness, pole_radius, 0.0, meshSize)
p8 = factory.addPoint(pole_gap/2.0, pole_radius, 0.0, meshSize)

print(p1)
print(p2)

lines = []
lines.append(factory.addLine(p1, p2))
lines.append(factory.addLine(p2, p3))
lines.append(factory.addLine(p3, p4))
lines.append(factory.addLine(p4, p5))
lines.append(factory.addLine(p5, p6))
lines.append(factory.addLine(p6, p7))
lines.append(factory.addLine(p7, p8))
lines.append(factory.addLine(p8, p1))


loop = factory.addCurveLoop(lines)
yoke = factory.addPlaneSurface([loop])

out = factory.cut([(2,rect)],[(2,current)], removeTool = False)
rect = out[0][0][1]
print(rect)

#out = factory.fragment([(2,rect)],[(2,yoke)])
out = factory.cut([(2,rect)],[(2,yoke)], removeTool = False)
print(rect)
print(yoke)
print(out)

tag_0 = out[0][0][1]
print(tag_0)
tag_1 = out[0][1][1]
print(tag_1)
tags = [tag_0, tag_1]


#gmsh.model.addPhysicalGroup(2, tags, tag = 1, name = 'vacuum')

#excitation = factory.addRectangle(0.0, 0.0, 0.0, box_size_x, box_size_y)

factory.synchronize()


gmsh.model.mesh.generate(2)

print(current)
print(yoke)
gmsh.model.addPhysicalGroup(2, tags, tag = 1, name = 'vacuum')
gmsh.model.addPhysicalGroup(2, [current], tag = 2, name = 'copper')
gmsh.model.addPhysicalGroup(2, [yoke], tag = 3, name = 'iron')


gmsh.write("iron_yoke_2d_001.msh")

# Run the GMSH GUI to visualize the mesh (comment out if you don't want to use the GUI)
gmsh.fltk.run()

# Finalize the GMSH API
gmsh.finalize()

