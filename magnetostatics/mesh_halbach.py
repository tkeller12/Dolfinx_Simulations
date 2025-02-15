import numpy as np

import gmsh

gmsh.initialize()

gmsh.model.add("2D_Rectangle")

factory = gmsh.model.occ

DEFAULT_MESH_SIZE = 0.01
FINE_MESH_SIZE = 0.005
magnet_mesh_size = 0.005
box_size = 1.0
r = 0.5
magnet_size = 0.070

total_magnets = 16

if (total_magnets % 4) !=0:
    raise ValueError('total magnets must be divisible by 4')

def addRectangle(x, y, z, dx, dy, theta = 0., theta_2 = 0., meshSize = DEFAULT_MESH_SIZE):
    if type(meshSize) is list:
        p1 = factory.addPoint(x, y, z, meshSize[0])
        p2 = factory.addPoint(x+dx, y, z, meshSize[1])
        p3 = factory.addPoint(x+dx, y+dy, z, meshSize[2])
        p4 = factory.addPoint(x, y+dy, z, meshSize[3])
    else:
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

    factory.rotate([(2,rect)],0, 0, 0, 0, 0, 1, theta * np.pi / 180.)

    center_of_mass = gmsh.model.occ.getCenterOfMass(2, rect)
    center_x = center_of_mass[0]
    center_y = center_of_mass[1]
    factory.rotate([(2,rect)],center_x, center_y, 0, 0, 0, 1, theta_2 * np.pi / 180.)


    return rect




boundary = addRectangle(0,0,0,box_size, box_size, meshSize = [FINE_MESH_SIZE, DEFAULT_MESH_SIZE, DEFAULT_MESH_SIZE, DEFAULT_MESH_SIZE])

magnets = int((total_magnets / 4) + 1)
index = range(magnets)
#print(index)
#print(index[-1])

magnet_tags = []
M_angle = []
for each in index:
    print(each)
    theta = each*(90/(magnets-1))
    M_angle.append(theta*2.0)
    print(theta)
    if (each == 0):
        rect = addRectangle(r-magnet_size/2, 0, 0, magnet_size, magnet_size/2., theta = theta, theta_2 = theta, meshSize = magnet_mesh_size)
    elif each == index[-1]:
        rect = addRectangle(r-magnet_size/2, 0, 0, magnet_size, -magnet_size/2., theta = theta, theta_2 = 0, meshSize = magnet_mesh_size)
    else:
        rect = addRectangle(r-magnet_size/2, -magnet_size/2, 0, magnet_size, magnet_size, theta = theta, theta_2 = theta, meshSize = magnet_mesh_size)

    factory.cut([(2,boundary)], [(2, rect)], removeTool = False)
    magnet_tags.append(rect)
print(M_angle)
print(magnet_tags)
for ix, M_value in enumerate(M_angle):
    print(magnet_tags[ix], M_value)
    print(np.sin(M_value * np.pi / 180.), np.cos(M_value * np.pi / 180.))

factory.addPoint(0,0,0)
factory.synchronize()


gmsh.model.mesh.generate(2)


gmsh.write("halbach_2d_001.msh")

# Run the GMSH GUI to visualize the mesh (comment out if you don't want to use the GUI)
gmsh.fltk.run()

# Finalize the GMSH API
gmsh.finalize()



